#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/JSON.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/PassBuilder.h"
#include <map>
#include <set>
#include <vector>
#include <string>
#include <sstream>

using namespace llvm;

namespace {

// Helper to get bitwidth from a value (capped at 32)
unsigned getBitWidth(Value *V) {
  unsigned width = 0;
  if (auto *Ty = dyn_cast<IntegerType>(V->getType())) {
    width = Ty->getBitWidth();
  } else if (V->getType()->isFloatTy()) {
    width = 32;
  } else if (V->getType()->isDoubleTy()) {
    width = 64;
  } else if (V->getType()->isPointerTy()) {
    width = 64;
  }
  // Cap at 32
  return width > 32 ? 32 : width;
}

// Helper to map operation to functional unit type
std::string mapToFunctionalUnit(const std::string& opcode) {
  // ALU operations (includes mul and div)
  if (opcode == "add" || opcode == "sub" || opcode == "mul" || 
      opcode == "div" || opcode == "mod" || opcode == "and" || 
      opcode == "or" || opcode == "xor" || opcode == "shl" || 
      opcode == "lshr" || opcode == "ashr" || opcode == "icmp") {
    return "alu";
  }
  
  // FP operations
  if (opcode == "fadd" || opcode == "fsub" || opcode == "fmul" || 
      opcode == "fdiv" || opcode == "fma") {
    return "fpu";
  }
  
  // Memory operations
  if (opcode == "load" || opcode == "store") {
    return "mem";
  }
  
  // Constants and special
  if (opcode == "const") {
    return "const";
  }
  
  // Default
  return "alu";
}

// Helper to map LLVM instruction to DORA operation type
std::string mapToDORAOperation(Instruction *I) {
  switch (I->getOpcode()) {
    // Arithmetic
    case Instruction::Add:
    case Instruction::FAdd:
      return I->getType()->isFloatingPointTy() ? "fadd" : "add";
    case Instruction::Sub:
    case Instruction::FSub:
      return I->getType()->isFloatingPointTy() ? "fsub" : "sub";
    case Instruction::Mul:
    case Instruction::FMul:
      return I->getType()->isFloatingPointTy() ? "fmul" : "mul";
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::FDiv:
      return I->getType()->isFloatingPointTy() ? "fdiv" : "div";
    case Instruction::URem:
    case Instruction::SRem:
      return "mod";
    
    // Logical
    case Instruction::And:
      return "and";
    case Instruction::Or:
      return "or";
    case Instruction::Xor:
      return "xor";
    
    // Shift
    case Instruction::Shl:
      return "shl";
    case Instruction::LShr:
      return "lshr";
    case Instruction::AShr:
      return "ashr";
    
    // Memory
    case Instruction::Load:
      return "load";
    case Instruction::Store:
      return "store";
    case Instruction::GetElementPtr:
      return "gep";
    
    // Comparison
    case Instruction::ICmp:
    case Instruction::FCmp:
      return "icmp";
    
    // Conversion
    case Instruction::Trunc:
      return "trunc";
    case Instruction::ZExt:
      return "nop";
    case Instruction::SExt:
      return "nop";
    case Instruction::FPToUI:
    case Instruction::FPToSI:
      return "nop";
    case Instruction::UIToFP:
    case Instruction::SIToFP:
      return "nop";
    case Instruction::PtrToInt:
      return "nop";
    case Instruction::IntToPtr:
      return "nop";
    case Instruction::BitCast:
      return "nop";
    
    // Control flow
    case Instruction::PHI:
      return "phi";
    case Instruction::Select:
      return "select";
    case Instruction::Br:
      return "br";
    
    // Call
    case Instruction::Call: {
      auto *CI = dyn_cast<CallInst>(I);
      if (auto *Callee = CI->getCalledFunction()) {
        if (Callee->getName().contains("fmuladd") || Callee->getName().contains("llvm.fma")) {
          return "fma";
        } else if (Callee->getName().contains("sqrt")) {
          return "sqrt";
        }
      }
      return "call";
    }
    
    default:
      return "unknown";
  }
}

// Extract constants from an instruction
std::vector<std::pair<std::string, int64_t>> extractConstants(Instruction *I) {
  std::vector<std::pair<std::string, int64_t>> constants;
  
  for (unsigned i = 0; i < I->getNumOperands(); i++) {
    Value *Op = I->getOperand(i);
    if (auto *CI = dyn_cast<ConstantInt>(Op)) {
      constants.push_back({"int", CI->getSExtValue()});
    } else if (auto *CF = dyn_cast<ConstantFP>(Op)) {
      // Store FP constant as bitcast to int64
      if (CF->getType()->isFloatTy()) {
        constants.push_back({"float", 
          static_cast<int64_t>(CF->getValueAPF().bitcastToAPInt().getZExtValue())});
      } else if (CF->getType()->isDoubleTy()) {
        constants.push_back({"double", 
          static_cast<int64_t>(CF->getValueAPF().bitcastToAPInt().getZExtValue())});
      }
    }
  }
  
  return constants;
}

// Lower GEP instructions to arithmetic operations
struct LowerGEPPass : public PassInfoMixin<LowerGEPPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    bool Changed = false;
    std::vector<GetElementPtrInst*> GEPsToLower;
    
    // Collect GEP instructions
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
          GEPsToLower.push_back(GEP);
        }
      }
    }
    
    // Lower each GEP to arithmetic
    for (auto *GEP : GEPsToLower) {
      IRBuilder<> Builder(GEP);
      Value *Base = GEP->getPointerOperand();
      Value *Result = Builder.CreatePtrToInt(Base, Builder.getInt64Ty());
      
      // Calculate offset
      for (unsigned i = 1; i < GEP->getNumOperands(); i++) {
        Value *Idx = GEP->getOperand(i);
        unsigned ElemSize = GEP->getSourceElementType()->getPrimitiveSizeInBits() / 8;
        
        // Always create explicit constant for element size and use mul
        Value *ElemSizeVal = Builder.getInt64(ElemSize);
        Value *IdxExt = Builder.CreateSExtOrTrunc(Idx, Builder.getInt64Ty());
        Value *Offset = Builder.CreateMul(IdxExt, ElemSizeVal, "gep.offset");
        Result = Builder.CreateAdd(Result, Offset, "gep.add");
      }
      
      Result = Builder.CreateIntToPtr(Result, GEP->getType());
      GEP->replaceAllUsesWith(Result);
      GEP->eraseFromParent();
      Changed = true;
    }
    
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

// DFG Extraction Pass - Extract DFG between __kernel_region_start/end
struct DFGExtractionPass : public PassInfoMixin<DFGExtractionPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    // Find kernel region markers
    CallInst *RegionStart = nullptr;
    CallInst *RegionEnd = nullptr;
    
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          if (auto *Callee = CI->getCalledFunction()) {
            if (Callee->getName() == "__kernel_region_start") {
              RegionStart = CI;
            } else if (Callee->getName() == "__kernel_region_end") {
              RegionEnd = CI;
            }
          }
        }
      }
    }
    
    if (!RegionStart || !RegionEnd) {
      errs() << "No kernel region found in function " << F.getName() << "\n";
      return PreservedAnalyses::all();
    }
    
    // Extract instructions between markers
    std::vector<Instruction*> KernelInsts;
    bool InRegion = false;
    
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (&I == RegionStart) {
          InRegion = true;
          continue;
        }
        if (&I == RegionEnd) {
          InRegion = false;
          break;
        }
        if (InRegion) {
          KernelInsts.push_back(&I);
        }
      }
    }
    
    // Build DFG JSON
    json::Object DFG;
    json::Array Nodes;
    json::Array Edges;
    
    std::map<Value*, std::string> ValueToNodeID;
    unsigned NodeID = 0;
    
    // First pass: create nodes for constants and inputs
    std::map<Value*, std::string> ConstantNodes;
    for (Instruction *I : KernelInsts) {
      for (unsigned i = 0; i < I->getNumOperands(); i++) {
        Value *Op = I->getOperand(i);
        
        // Skip if already handled
        if (ValueToNodeID.find(Op) != ValueToNodeID.end()) continue;
        
        // Skip if it's an instruction in the kernel region
        if (auto *OpInst = dyn_cast<Instruction>(Op)) {
          bool inRegion = std::find(KernelInsts.begin(), KernelInsts.end(), OpInst) != KernelInsts.end();
          if (inRegion) continue;
        }
        
        // Handle constants
        if (isa<Constant>(Op) && !isa<Function>(Op)) {
          if (ConstantNodes.find(Op) == ConstantNodes.end()) {
            std::string NodeIDStr = std::to_string(NodeID++);
            ConstantNodes[Op] = NodeIDStr;
            ValueToNodeID[Op] = NodeIDStr;
            
            json::Object Node;
            Node["id"] = NodeIDStr;
            Node["opcode"] = "const";
            Node["fu"] = "const";
            Node["bitwidth"] = getBitWidth(Op);
            
            // Store the constant value
            if (auto *CI = dyn_cast<ConstantInt>(Op)) {
              Node["value"] = CI->getSExtValue();
            } else if (auto *CF = dyn_cast<ConstantFP>(Op)) {
              if (CF->getType()->isFloatTy()) {
                Node["value"] = static_cast<int64_t>(CF->getValueAPF().bitcastToAPInt().getZExtValue());
              } else if (CF->getType()->isDoubleTy()) {
                Node["value"] = static_cast<int64_t>(CF->getValueAPF().bitcastToAPInt().getZExtValue());
              }
            }
            
            Nodes.push_back(std::move(Node));
          }
        } else {
          // Handle inputs (values from outside the kernel region)
          std::string NodeIDStr = std::to_string(NodeID++);
          ValueToNodeID[Op] = NodeIDStr;
          
          json::Object Node;
          Node["id"] = NodeIDStr;
          Node["opcode"] = "input";
          Node["fu"] = "input";
          Node["bitwidth"] = getBitWidth(Op);
          
          Nodes.push_back(std::move(Node));
        }
      }
    }
    
    // Second pass: create nodes for instructions (skip NOPs)
    for (Instruction *I : KernelInsts) {
      std::string opcode = mapToDORAOperation(I);
      
      // Skip NOP instructions and wire through
      if (opcode == "nop") {
        // Wire through: map this instruction to its operand
        if (I->getNumOperands() > 0) {
          Value *Op = I->getOperand(0);
          if (ValueToNodeID.find(Op) != ValueToNodeID.end()) {
            ValueToNodeID[I] = ValueToNodeID[Op];
          }
        }
        continue;
      }
      
      std::string NodeIDStr = std::to_string(NodeID++);
      ValueToNodeID[I] = NodeIDStr;
      
      json::Object Node;
      Node["id"] = NodeIDStr;
      Node["opcode"] = opcode;
      Node["fu"] = mapToFunctionalUnit(opcode);
      Node["bitwidth"] = getBitWidth(I);
      
      Nodes.push_back(std::move(Node));
    }
    
    // Create edges for dataflow dependencies and build pred/succ maps
    std::map<std::string, std::vector<std::string>> predecessors;
    std::map<std::string, std::vector<std::string>> successors;
    
    for (Instruction *I : KernelInsts) {
      if (ValueToNodeID.find(I) == ValueToNodeID.end()) continue;
      
      // Skip if this was a NOP that got wired through
      if (mapToDORAOperation(I) == "nop") continue;
      
      std::string DestID = ValueToNodeID[I];
      
      for (unsigned i = 0; i < I->getNumOperands(); i++) {
        Value *Op = I->getOperand(i);
        
        // Create edges for both instructions and constants
        if (ValueToNodeID.find(Op) != ValueToNodeID.end()) {
          std::string SourceID = ValueToNodeID[Op];
          
          json::Object Edge;
          Edge["source"] = SourceID;
          Edge["dest"] = DestID;
          Edge["bitwidth"] = getBitWidth(Op);
          Edges.push_back(std::move(Edge));
          
          // Track predecessors and successors
          predecessors[DestID].push_back(SourceID);
          successors[SourceID].push_back(DestID);
        }
      }
    }
    
    // Add pred/succ to nodes
    for (auto& NodeVal : Nodes) {
      auto* NodeObj = NodeVal.getAsObject();
      if (!NodeObj) continue;
      
      auto IdVal = NodeObj->getString("id");
      if (!IdVal) continue;
      std::string nodeId = IdVal->str();
      
      // Add predecessors
      json::Array PredArray;
      if (predecessors.find(nodeId) != predecessors.end()) {
        for (const auto& pred : predecessors[nodeId]) {
          PredArray.push_back(pred);
        }
      }
      NodeObj->insert({"pred", std::move(PredArray)});
      
      // Add successors
      json::Array SuccArray;
      if (successors.find(nodeId) != successors.end()) {
        for (const auto& succ : successors[nodeId]) {
          SuccArray.push_back(succ);
        }
      }
      NodeObj->insert({"succ", std::move(SuccArray)});
    }
    
    // Build JSON manually to control field order: function, nodes, edges
    std::string JSONStr = "{\n";
    JSONStr += "  \"function\": \"" + F.getName().str() + "\",\n";
    
    // Add nodes array
    JSONStr += "  \"nodes\": [\n";
    for (size_t i = 0; i < Nodes.size(); ++i) {
      std::string nodeStr;
      raw_string_ostream NOS(nodeStr);
      NOS << formatv("{0:2}", Nodes[i]);
      NOS.flush();
      
      // Indent the node object
      std::string indented;
      for (char c : nodeStr) {
        if (c == '\n') {
          indented += "\n    ";
        } else {
          indented += c;
        }
      }
      JSONStr += "    " + indented;
      if (i < Nodes.size() - 1) {
        JSONStr += ",\n";
      } else {
        JSONStr += "\n";
      }
    }
    JSONStr += "  ],\n";
    
    // Add edges array
    JSONStr += "  \"edges\": [\n";
    for (size_t i = 0; i < Edges.size(); ++i) {
      std::string edgeStr;
      raw_string_ostream EOS(edgeStr);
      EOS << formatv("{0:2}", Edges[i]);
      EOS.flush();
      
      // Indent the edge object
      std::string indented;
      for (char c : edgeStr) {
        if (c == '\n') {
          indented += "\n    ";
        } else {
          indented += c;
        }
      }
      JSONStr += "    " + indented;
      if (i < Edges.size() - 1) {
        JSONStr += ",\n";
      } else {
        JSONStr += "\n";
      }
    }
    JSONStr += "  ]\n";
    JSONStr += "}\n";
    
    // Get source filename from module (without path and extension)
    std::string sourceFile = F.getParent()->getSourceFileName();
    size_t lastSlash = sourceFile.find_last_of("/\\");
    if (lastSlash != std::string::npos) {
      sourceFile = sourceFile.substr(lastSlash + 1);
    }
    size_t lastDot = sourceFile.find_last_of(".");
    if (lastDot != std::string::npos) {
      sourceFile = sourceFile.substr(0, lastDot);
    }
    
    std::string outputFilename = sourceFile + "_dfg.json";
    
    // Write to file
    std::error_code EC;
    raw_fd_ostream OutFile(outputFilename, EC);
    if (!EC) {
      OutFile << JSONStr;
      OutFile.close();
      errs() << "DFG written to " << outputFilename << "\n";
    } else {
      errs() << "Error writing DFG file: " << EC.message() << "\n";
    }
    
    return PreservedAnalyses::all();
  }
};

} // anonymous namespace

// Register passes with new pass manager
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION, "DoraMapperPasses", LLVM_VERSION_STRING,
    [](PassBuilder &PB) {
      PB.registerPipelineParsingCallback(
        [](StringRef Name, FunctionPassManager &FPM,
           ArrayRef<PassBuilder::PipelineElement>) {
          if (Name == "lower-gep") {
            FPM.addPass(LowerGEPPass());
            return true;
          }
          if (Name == "extract-dfg") {
            FPM.addPass(DFGExtractionPass());
            return true;
          }
          return false;
        }
      );
    }
  };
}
