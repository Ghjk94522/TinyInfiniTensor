#include "core/graph.h"
#include "operators/transpose.h"
#include "operators/matmul.h"
#include <algorithm>
#include <numeric>
#include <queue>

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // =================================== 作业 ===================================

        /// opt rule 1: del the redundant transpose ops
        if (!this->sorted)
            topo_sort();

        std::vector<Operator> opToRemove;
        std::vector<Tensor> tensorToRemove;

        for (auto op : ops) {
            if (op->getOpType() == OpType::Transpose) {
                std::vector<Operator> toRemovedSuccessor;
                for (auto successor : op->getSuccessors()) {
                    if (successor->getOpType() == OpType::Transpose) {
                        if (op->getOutput() == successor->getInputs(0)) {
                            // now check if the permute same
                            auto transOp = std::dynamic_pointer_cast<TransposeObj>(op);
                            auto transOpSuc = std::dynamic_pointer_cast<TransposeObj>(successor);
                            auto permA = transOp->getPermute();
                            auto permB = transOpSuc->getPermute();
                            if (permA != permB)
                                continue;

                            auto oriTensor = op->getInputs(0);
                            auto dstTensor = successor->getOutput();
                            auto midTensor = op->getOutput();

                            // replace all uses with the oriTensor
                            for (auto tarOp : dstTensor->getTargets()) {
                                auto tarInputs = tarOp->getInputs();
                                for (size_t i = 0; i < tarInputs.size(); i++) {
                                    if (tarInputs[i] == dstTensor) {
                                        tarOp->setInput(i, oriTensor);
                                        oriTensor->addTarget(tarOp);
                                        tarOp->removePredecessors(successor);
                                        if (auto pred = oriTensor->getSource())
                                            tarOp->addPredecessors(pred);
                                    }
                                }
                            }

                            // record the ops and tensors to be removed
                            opToRemove.emplace_back(successor);
                            toRemovedSuccessor.emplace_back(successor);
                            tensorToRemove.emplace_back(dstTensor);
                            tensorToRemove.emplace_back(midTensor);
                        }
                    }
                }

                // remove the successor from the src transpose op
                for (auto successor : toRemovedSuccessor) {
                    op->removeSuccessors(successor);
                }
                if (op->getSuccessors().empty())
                    opToRemove.emplace_back(op);
            }
        }

        // remove the useless transpose op and its relationship
        for (auto op : opToRemove) {
            for (auto opInput : op->getInputs()) {
                opInput->removeTarget(op);
            }

            removeOperator(op);
        }
        opToRemove.clear();

        // remove the useless tensor
        for (auto tensor : tensorToRemove) {
            removeTensor(tensor);
        }
        tensorToRemove.clear();

        /// opt rule 2: fuse transpose and matmul op when the input of matmul has trans attr
        for (auto op : ops) {
            if (op->getOpType() == OpType::MatMul) {
                auto matmulOp = as<MatmulObj>(op);
                for (size_t f = 0; f < 2; f++) {
                    // which means 0 => A, 1 => B
                    auto inputTensor = op->getInputs(f);
                    if(auto inputOp = inputTensor->getSource()) {
                        if (inputOp->getOpType() == OpType::Transpose) {
                            // check the transpose op
                            auto transposeOp = as<TransposeObj>(inputOp);
                            auto perm = transposeOp->getPermute();
                            auto isAlmostPerm = [] (std::vector<int>& v) -> bool {
                                size_t n = v.size();
                                for (size_t i = 0; i < n - 2; i++) {
                                    if (v[i] != static_cast<int>(i)) 
                                        return false;
                                }
                                if (v[n - 2] != static_cast<int>(n - 1) || v[n - 1] != static_cast<int>(n - 2))
                                    return false;
                                return true;
                            };

                            if (!isAlmostPerm(perm))
                                continue;
                            
                            // remove the transpose and set new operand for matmul op
                            auto oriTensor = inputOp->getInputs(0);
                            auto dstTensor = inputOp->getOutput();
                            matmulOp->setInput(f, oriTensor);
                            oriTensor->addTarget(op);
                            if (f == 0) {
                                matmulOp->setTransA(!matmulOp->getTransA());
                            } else {
                                matmulOp->setTransB(!matmulOp->getTransB());
                            }
                            matmulOp->removePredecessors(inputOp);
                            if (auto oriOp = oriTensor->getSource())
                                matmulOp->addPredecessors(oriOp);
                            inputOp->removeSuccessors(op);
                            if (inputOp->getSuccessors().empty())
                                opToRemove.push_back(inputOp);
                            tensorToRemove.push_back(dstTensor);
                        }
                    }
                }
            }
        }

        for (auto op : opToRemove) {
            for (auto opInput : op->getInputs()) {
                opInput->removeTarget(op);
            }

            removeOperator(op);
        }
        
        // remove all useless tensors
        for (auto tensor : tensorToRemove) {
            removeTensor(tensor);
        }

        topo_sort();
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        allocator.info();
        std::vector<size_t> offs;
        for (auto vec : tensors) {
            offs.push_back(allocator.alloc(vec->getBytes()));
        }

        auto *ptr = static_cast<char *>(allocator.getPtr());
        for (size_t i = 0; i < tensors.size(); i++) {
            auto blob = make_ref<BlobObj>(this->runtime, ptr + offs[i]);
            tensors[i]->setDataBlob(blob);
        }
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini