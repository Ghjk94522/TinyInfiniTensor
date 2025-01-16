#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
        block_table.emplace(0, 0xfffffff);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================

        size_t offs = 0;
        bool flag = false;
        for (auto& pair : block_table) {
            if (pair.second >= size) {
                offs = pair.first;
                flag = true;
                break;
            }
        }
        IT_ASSERT(flag, "cannot find a free mem block.");

        if (block_table[offs] - size > 0)
            block_table[offs + size] = block_table[offs] - size;
        auto iter = block_table.find(offs);
        if (iter == block_table.end()) 
            IT_ASSERT(false, "cannot find the block iter.");
        block_table.erase(iter);

        used += size;
        if (used > peak)
            peak = used;

        return offs;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        size_t candidate = addr + size;
        auto iter = block_table.find(candidate);
        if (iter != block_table.end()) {
            size_t candiSize = block_table[candidate];
            block_table[addr] = size + candiSize;
            block_table.erase(iter);
        } else {
            block_table[addr] = size;
        }

        used -= size;
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
