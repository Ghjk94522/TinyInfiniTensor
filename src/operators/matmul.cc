#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        Shape A = inputs[0]->getDims();
        Shape B = inputs[1]->getDims();
        Shape res(A.size(), 0);
        
        if (A.size() == 4) {
            res[0] = A[0];
            res[1] = A[1];
        } else if (A.size() == 3) {
            res[0] = A[0];
        } else if (A.size() != 2) {
            IT_ASSERT(false, "illegal shape in matmul op");
        }

        res[res.size() - 2] = getTransA() ? A[A.size() - 1] : A[A.size() - 2];
        res[res.size() - 1] = getTransB() ? B[B.size() - 2] : B[B.size() - 1];

        return std::vector<Shape> (1, std::move(res));
    }

} // namespace infini