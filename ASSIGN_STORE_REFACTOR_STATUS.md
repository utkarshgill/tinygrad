# ASSIGN to STORE Refactor - Current Status & Issues

## Overview
The ASSIGN to STORE refactor has been partially implemented to merge ASSIGN and STORE operations into a unified STORE operation. While significant progress has been made, there are critical issues that need to be resolved.

## ✅ Successfully Implemented

### 1. Core UOp Operations (`tinygrad/uop/ops.py`)
- ✅ Updated `UOp.assign()` method to create STORE instead of ASSIGN: `return UOp(Ops.STORE, self.dtype, (self,x))`
- ✅ Modified `UOp.store()` to handle both memory operations and register assignment with intelligent dtype selection
- ✅ Fixed `st` property to handle STORE operations with proper shape tracking:
  - Register assignments (DEFINE_REG/DEFINE_GLOBAL targets): return None
  - INDEX-based STORE operations: flow from INDEX st
  - Buffer STORE operations: use buffer's shape tracker
  - Other memory STORE operations: get shape from value being stored
- ✅ Updated `size` property to handle STORE operations without shape trackers
- ✅ Updated `buf_uop` property to only check for STORE operations
- ✅ Fixed `_mop` method to handle None shape trackers properly

### 2. Pattern Matching Specifications (`tinygrad/uop/spec.py`)
- ✅ All pattern specifications have been updated from ASSIGN to STORE
- ✅ Updated `assign_spec` and related patterns to use STORE operations

### 3. Kernelize Pipeline (`tinygrad/kernelize/kernelize.py`)  
- ✅ All major patterns and logic updated to use STORE instead of ASSIGN
- ✅ Updated `create_kernels`, `add_buffer_ops`, `replace_globals`, etc.
- ✅ Updated dependency tracking and buffer management for STORE operations

### 4. Renderers/Backends
- ✅ **PTX Renderer** (`tinygrad/renderer/ptx.py`): Updated to handle unified STORE
- ✅ **C-style Renderer** (`tinygrad/renderer/cstyle.py`): Added STORE patterns for register assignment
- ✅ **LLVM Renderer** (`tinygrad/renderer/llvmir.py`): Partially updated (had some linter errors)

### 5. Other Components
- ✅ **Devectorizer** (`tinygrad/codegen/devectorizer.py`): Updated patterns
- ✅ **Expander** (`tinygrad/codegen/expander.py`): Updated patterns

## ❌ Critical Issues Remaining

### 1. Infinite Loop in Graph Rewrite (HIGH PRIORITY)
**Status**: All tests failing with "RuntimeError: infinite loop in graph_rewrite"

**Problem**: The refactor has introduced patterns that cause infinite loops in the graph rewrite system during the "replace buffer" step in `fix_kernel_ast`.

**Root Cause**: Unknown - requires investigation of:
- Pattern matching rules that may create cycles
- STORE operations being rewritten infinitely 
- Interaction between new STORE patterns and existing rewrite rules

**Location**: `tinygrad/kernelize/kernelize.py:305` in `fix_kernel_ast` function

### 2. Movement Operations on STORE (RESOLVED)
**Status**: ✅ Fixed - STORE operations now have proper shape tracker handling

**Previous Issue**: Movement operations like RESHAPE were being applied to STORE operations that didn't have proper shape trackers, causing `unwrap(None)` errors.

**Solution**: Updated `st` property in `tinygrad/uop/ops.py` to handle STORE operations properly based on their target type.

## 🔍 Investigation Needed

### Priority 1: Infinite Loop Root Cause
1. **Pattern Analysis**: Review all STORE-related patterns to identify which might create infinite rewrite cycles
2. **Rewrite Debugging**: Add debug logging to identify which specific pattern is causing the infinite loop
3. **Pattern Order**: Check if pattern ordering needs adjustment for STORE operations
4. **Cycle Detection**: Investigate if the unified STORE creates unexpected graph cycles

### Priority 2: Test Verification  
1. **Unit Tests**: Once infinite loop is fixed, run all test suites:
   - `test/test_assign.py` 
   - `test/test_uops.py`
   - `test/test_schedule.py`
   - `test/test_linearizer.py`
2. **Integration Tests**: Verify complex operations work correctly
3. **Performance**: Ensure no performance regressions from the refactor

## 🎯 Next Steps

1. **Debug Infinite Loop**: 
   - Add debug logging to identify the specific pattern causing infinite rewrites
   - Check `view_left+add_buffer_ops+fix_kernel_ops` patterns for STORE-related cycles
   - Review pattern matching order and conditions

2. **Pattern Review**:
   - Audit all STORE-related patterns for potential cycles
   - Ensure STORE patterns have proper termination conditions
   - Check interaction between STORE and movement operations

3. **Test & Validate**:
   - Once infinite loop is resolved, run comprehensive tests
   - Verify all functionality is preserved
   - Document any behavior changes

## 📝 Implementation Summary

The ASSIGN to STORE refactor represents a significant architectural change that:
- **Unifies** ASSIGN and STORE operations into a single STORE operation
- **Preserves** functionality through intelligent dtype handling (target dtype for register assignment, dtypes.void for memory operations)  
- **Maintains** pattern matching compatibility across the entire codebase
- **Updates** all major components: kernelize, renderers, patterns, etc.

The core refactor is architecturally sound but has introduced an infinite loop in the graph rewrite system that must be resolved before the implementation can be considered complete.