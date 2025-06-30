# ASSIGN to STORE Refactor Plan

## Current State Analysis

### ASSIGN Usage:
- **Purpose**: Register/variable assignment 
- **Pattern**: `ASSIGN(target, value)` where target is `DEFINE_REG` or `DEFINE_GLOBAL`
- **Dtype**: Preserves target's dtype (`self.dtype`)
- **Method**: `def assign(self, x:UOp): return UOp(Ops.ASSIGN, self.dtype, (self,x))`

### STORE Usage:
- **Purpose**: Memory operations
- **Pattern**: `STORE(indexed_buffer, value, gate?)` where indexed_buffer is `INDEX(buffer, offset, mask?)`
- **Dtype**: Always `dtypes.void`
- **Method**: `def store(self, *src:UOp, **kwargs): return UOp(Ops.STORE, dtypes.void, (self,)+src, **kwargs)`

## Unification Strategy

The unified STORE operation will handle both cases:

1. **Memory operations**: `STORE(INDEX(buffer, offset), value, gate?)` - dtype=`dtypes.void`
2. **Register assignment**: `STORE(DEFINE_REG/DEFINE_GLOBAL, value)` - dtype=target's dtype

## Implementation Steps

### 1. Update Core UOp Methods
- [ ] Modify `UOp.assign()` to create STORE instead of ASSIGN
- [ ] Ensure STORE handles both indexed and direct targets

### 2. Update Pattern Specifications  
- [ ] `tinygrad/uop/spec.py`: Update patterns to recognize unified STORE
- [ ] Remove assign_spec patterns
- [ ] Update STORE patterns to handle both cases

### 3. Update Symbolic Rewriting
- [ ] `tinygrad/uop/symbolic.py`: Update ASSIGN patterns to use STORE

### 4. Update All Renderers/Backends
- [ ] `tinygrad/renderer/ptx.py`
- [ ] `tinygrad/renderer/cstyle.py` 
- [ ] `tinygrad/renderer/llvmir.py`
- [ ] `tinygrad/renderer/wgsl.py`
- [ ] `extra/backends/triton.py`
- [ ] Other backends in `extra/`

### 5. Update Codegen Pipeline
- [ ] `tinygrad/codegen/linearize.py`: Update BlockContext to handle unified STORE
- [ ] `tinygrad/codegen/devectorizer.py`: Update patterns
- [ ] `tinygrad/codegen/lowerer.py`
- [ ] `tinygrad/codegen/expander.py`

### 6. Update Core Operations
- [ ] `tinygrad/uop/__init__.py`: Remove ASSIGN from Ops enum
- [ ] `tinygrad/uop/ops.py`: Update all ASSIGN references

### 7. Update Tests
- [ ] Update all test files to use STORE instead of ASSIGN
- [ ] Ensure all functionality is preserved

### 8. Final Cleanup
- [ ] Remove all remaining ASSIGN references
- [ ] Update comments and documentation

## Key Considerations

1. **Dtype Handling**: STORE should use `dtypes.void` for memory ops, preserve target dtype for register assignment
2. **Pattern Matching**: All patterns must be updated to recognize both STORE variants
3. **Backend Compatibility**: Each backend must handle both register and memory STORE operations  
4. **Test Coverage**: Ensure all existing functionality is preserved

## Files to Modify

### Core UOp Files:
- `tinygrad/uop/__init__.py` - Remove ASSIGN from Ops
- `tinygrad/uop/ops.py` - Update assign method, pattern matching
- `tinygrad/uop/spec.py` - Update specification patterns
- `tinygrad/uop/symbolic.py` - Update symbolic rewriting

### Codegen Pipeline:
- `tinygrad/codegen/linearize.py`
- `tinygrad/codegen/devectorizer.py` 
- `tinygrad/codegen/lowerer.py`
- `tinygrad/codegen/expander.py`

### Renderers:
- `tinygrad/renderer/ptx.py`
- `tinygrad/renderer/cstyle.py`
- `tinygrad/renderer/llvmir.py`
- `tinygrad/renderer/wgsl.py`

### External Backends:
- `extra/backends/triton.py`
- Other backend files as needed

### Tests:
- All test files that reference ASSIGN operations