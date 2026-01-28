#include "cxconfig.h"
#include <cstdint>
#include <cstdlib>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fstream>
#include <iostream>

#include "llvm/Config/llvm-config.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/IRBuilder.h"

using namespace llvm;

int64_t find_line_number(const std::string &filepath, const std::string &target)
{
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        outs()<<"can not open file"<<filepath<<"\n";
        std::_Exit(0);
    }
    std::string line;
    int64_t lineNumber = 1;
    while (std::getline(file, line))
    {
        if (line.find(target) != std::string::npos)
        {
            return lineNumber;
        }
        lineNumber++;
    }
    return -1;
}

namespace
{
    class AFLCoverage : public PassInfoMixin<AFLCoverage>
    {
    public:
        AFLCoverage()
        {
        }
        PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
    };

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo()
{
    return {LLVM_PLUGIN_API_VERSION, "AFLCoverage", "v0.1", [](PassBuilder &PB)
            {
                PB.registerOptimizerEarlyEPCallback([](ModulePassManager &MPM, OptimizationLevel OL)
                                                    { MPM.addPass(AFLCoverage()); });
            }};
}

PreservedAnalyses AFLCoverage::run(Module &M, ModuleAnalysisManager &MAM)
{

    LLVMContext &C = M.getContext();

    IntegerType *Int8Ty = IntegerType::getInt8Ty(C);
    IntegerType *Int32Ty = IntegerType::getInt32Ty(C);
    IntegerType *Int64Ty = IntegerType::getInt64Ty(C);

    uint32_t map_size = MAP_SIZE;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, (map_size - 1)); //  generage randnum [0,(map_size - 1)]

    FunctionType *cxfunc1_type = FunctionType::get(Type::getInt64Ty(C), {}, false);
    FunctionCallee cxfunc1 = M.getOrInsertFunction("cxfunc1", cxfunc1_type);
    FunctionType *cxfunc2_type = FunctionType::get(Type::getInt64Ty(C), {}, false);
    FunctionCallee cxfunc2 = M.getOrInsertFunction("cxfunc2", cxfunc2_type);
    //  debug test
    FunctionType *cxprintf1_type = FunctionType::get(Type::getVoidTy(C), {PointerType::get(Int8Ty, 0)}, false);
    FunctionCallee cxprintf1 = M.getOrInsertFunction("cxprintf1", cxprintf1_type);

    for (auto &F : M)
    {
        if (F.isDeclaration())
        {
            continue;
        }
        if (F.size() < 2)
        {
            continue;
        }
        if (F.getSubprogram())
        {
            StringRef filename = F.getSubprogram()->getFilename();
            if (filename.starts_with("/usr/") || filename.contains("third_party/"))
            {
                continue;
            }
            BasicBlock &entryBB = F.getEntryBlock();
            BasicBlock::iterator IP = entryBB.getFirstInsertionPt();
            IRBuilder<> IRB(&(*IP));

            //  debug test
            // std::string filename_str = filename.str();
            // std::string funcname_str = F.getName().str();
            // std::string cxprint_str1 = filename_str + "-:-" + funcname_str;
            // Value *str_ptr = IRB.CreateGlobalStringPtr(cxprint_str1);
            // IRB.CreateCall(cxprintf1, {str_ptr});
            
            //  trace bits addr
            Value *array_addr = IRB.CreateCall(cxfunc1, {});
            Value *base_ptr = IRB.CreateIntToPtr(array_addr, Int8Ty->getPointerTo());
            Value *prevbb_ptr = IRB.CreateGEP(Int8Ty, base_ptr, ConstantInt::get(Int32Ty, map_size));
            Value *prevbb_ptr_as_uint32t = IRB.CreateBitCast(prevbb_ptr, Int32Ty->getPointerTo());
            //  generate cur loc
            uint32_t entry_cur_loc = dist(gen);
            ConstantInt *entry_cur_loc_const = ConstantInt::get(Int32Ty, entry_cur_loc);
            //  load prev bb loc
            LoadInst *entry_prevbb = IRB.CreateLoad(IRB.getInt32Ty(), prevbb_ptr_as_uint32t);
            entry_prevbb->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, std::nullopt));
            Value *entry_prevbb_casted = IRB.CreateZExt(entry_prevbb, IRB.getInt32Ty());
            //  update bitmap
            Value *entry_target_ptr = IRB.CreateGEP(Int8Ty, base_ptr, IRB.CreateXor(entry_prevbb_casted, entry_cur_loc_const));
            LoadInst *entry_counter = IRB.CreateLoad(IRB.getInt8Ty(), entry_target_ptr);
            entry_counter->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, std::nullopt));
            Value *entry_incr = IRB.CreateAdd(entry_counter, ConstantInt::get(Int8Ty, 1));
            IRB.CreateStore(entry_incr, entry_target_ptr)->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, std::nullopt));
            //  set prev bb loc
            StoreInst *entry_store1 = IRB.CreateStore(ConstantInt::get(Int32Ty, entry_cur_loc >> 1), prevbb_ptr_as_uint32t);
            entry_store1->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, std::nullopt));
            for (auto &BB : F)
            {
                if (&BB == &F.getEntryBlock())
                {
                    continue;
                }
                if (BB.empty())
                {
                    continue;
                }
                IP = BB.getFirstInsertionPt();
                IRB.SetInsertPoint(&(*IP));
                uint32_t cur_loc = dist(gen);
                ConstantInt *cur_loc_const = ConstantInt::get(Int32Ty, cur_loc);
                LoadInst *prevbb = IRB.CreateLoad(IRB.getInt32Ty(), prevbb_ptr_as_uint32t);
                prevbb->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, std::nullopt));
                Value *prevbb_casted = IRB.CreateZExt(prevbb, IRB.getInt32Ty());
                Value *target_ptr = IRB.CreateGEP(Int8Ty, base_ptr, IRB.CreateXor(prevbb_casted, cur_loc_const));
                LoadInst *counter = IRB.CreateLoad(IRB.getInt8Ty(), target_ptr);
                counter->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, std::nullopt));
                Value *incr = IRB.CreateAdd(counter, ConstantInt::get(Int8Ty, 1));
                IRB.CreateStore(incr, target_ptr)->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, std::nullopt));
                StoreInst *store1 = IRB.CreateStore(ConstantInt::get(Int32Ty, cur_loc >> 1), prevbb_ptr_as_uint32t);
                store1->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, std::nullopt));
            }
        }
    }

    for (auto &F : M)
    {
        if (F.isDeclaration())
        {
            continue;
        }
        if (F.size() < 2)
        {
            continue;
        }
        if (F.getSubprogram())
        {
            StringRef filename = F.getSubprogram()->getFilename();
            if (filename.starts_with("/usr/") || filename.contains("third_party/"))
            {
                continue;
            }
            StringRef funcName = F.getName();
            std::string target = filename.str()+":"+funcName.str();
            std::string filepath = "/home/chenxu/fuzz/pytorch_fun_info_uniq.txt";
            int64_t ret = find_line_number(filepath, target);
            if (ret > 0)
            {
                BasicBlock &entryBB = F.getEntryBlock();
                BasicBlock::iterator IP = entryBB.getFirstInsertionPt();
                IRBuilder<> IRB(&(*IP));
                Value *array_addr = IRB.CreateCall(cxfunc2, {});
                Value *base_ptr = IRB.CreateIntToPtr(array_addr, Int8Ty->getPointerTo());
                Value *target_ptr = IRB.CreateGEP(Int8Ty, base_ptr, ConstantInt::get(Int32Ty, ret*4));
                Value *target_ptr_as_uint32t = IRB.CreateBitCast(target_ptr, Int32Ty->getPointerTo());
                uint32_t entry_cur_loc = dist(gen);
                ConstantInt *entry_cur_loc_const = ConstantInt::get(Int32Ty, entry_cur_loc);
                LoadInst *entry_exec_hash = IRB.CreateLoad(IRB.getInt32Ty(), target_ptr_as_uint32t);
                entry_exec_hash->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, std::nullopt));
                Value *entry_exec_hash_casted = IRB.CreateZExt(entry_exec_hash, IRB.getInt32Ty());
                Value *entry_cur_exec_hash = IRB.CreateXor(entry_exec_hash_casted, entry_cur_loc_const);
                StoreInst *entry_store1 = IRB.CreateStore(entry_cur_exec_hash, target_ptr_as_uint32t);
                entry_store1->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, std::nullopt));
                for (auto &BB : F)
                {
                    if (&BB == &F.getEntryBlock())
                    {
                        continue;
                    }
                    if (BB.empty())
                    {
                        continue;
                    }
                    IP = BB.getFirstInsertionPt();
                    IRB.SetInsertPoint(&(*IP));
                    uint32_t cur_loc = dist(gen);
                    ConstantInt *cur_loc_const = ConstantInt::get(Int32Ty, cur_loc);
                    LoadInst *exec_hash = IRB.CreateLoad(IRB.getInt32Ty(), target_ptr_as_uint32t);
                    exec_hash->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, std::nullopt));
                    Value *exec_hash_casted = IRB.CreateZExt(exec_hash, IRB.getInt32Ty());
                    Value *cur_exec_hash = IRB.CreateXor(exec_hash_casted, cur_loc_const);
                    StoreInst *store1 = IRB.CreateStore(cur_exec_hash, target_ptr_as_uint32t);
                    store1->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, std::nullopt));
                }
            }
        }
    }
    return PreservedAnalyses();
}