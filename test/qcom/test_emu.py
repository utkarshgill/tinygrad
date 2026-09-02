import struct, pytest
from test.mockgpu.qcom.emu import IR3Instruction, IR3Register, IR3Source, WaveState, decode_ir3, execute_instruction, execute_instructions

def f32_bits(value: float) -> int:
  return struct.unpack('<I', struct.pack('<f', value))[0]

def test_decode_add():
  expected = IR3Instruction(
    name='add.f',
    dst=IR3Register('r', 2, 3),
    srcs=(
      IR3Source(IR3Register('r', 2, 3)),
      IR3Source(IR3Register('r', 4, 0)),
    ),
    sy=True,
  )
  assert decode_ir3(bytes.fromhex('0b0010000b001050')) == [expected]

def test_decode_add_half_constant():
  expected = IR3Instruction(
    name='add.f',
    dst=IR3Register('hr', 0, 2),
    srcs=(
      IR3Source(IR3Register('hr', 1, 2), repeat=True),
      IR3Source(IR3Register('hc', 8, 0), repeat=True, negate=True),
    ),
    repeat=1,
  )
  assert decode_ir3(bytes.fromhex('0600205002090840')) == [expected]

def test_decode_integer_immediate():
  expected = IR3Instruction(
    name='sub.u',
    dst=IR3Register('hr', 1, 2),
    srcs=(
      IR3Source(1),
      IR3Source(IR3Register('hr', 1, 2)),
    ),
  )
  assert decode_ir3(bytes.fromhex('0120060006004042')) == [expected]

def test_decode_float_immediate():
  expected = IR3Instruction(
    name='mul.f',
    dst=IR3Register('r', 6, 0),
    srcs=(
      IR3Source(IR3Register('r', 6, 0)),
      IR3Source(0.3010300099849701),
    ),
    ss=True,
  )
  assert decode_ir3(bytes.fromhex('1800092818107040')) == [expected]

def test_decode_add_with_delay_slots():
  expected = IR3Instruction(
    name='add.f',
    dst=IR3Register('r', 0, 2),
    srcs=(
      IR3Source(IR3Register('r', 0, 2)),
      IR3Source(IR3Register('r', 1, 3)),
    ),
    sy=True,
    nop=3,
  )
  assert decode_ir3(bytes.fromhex('0200070002081850')) == [expected]

def test_full_register_storage():
  state = WaveState(2, 64)
  register = IR3Register('r', 2, 1)

  state.write_r(register, 0, 0x12345678)
  state.write_r(register, 1, 0xabcdef01)

  assert state.read_r(register, 0) == 0x12345678
  assert state.read_r(register, 1) == 0xabcdef01

def test_full_register_storage_rejects_invalid_coordinates():
  state = WaveState(2, 64)

  with pytest.raises(ValueError):
    state.read_r(IR3Register('r', 2, 4), 0)

  with pytest.raises(ValueError):
    state.read_r(IR3Register('r', 2, 0), 2)

def test_execute_add():
  instruction, = decode_ir3(bytes.fromhex('0b0010000b001050'))
  state = WaveState(2, 64)

  state.write_r(IR3Register('r', 2, 3), 0, f32_bits(2.0))
  state.write_r(IR3Register('r', 4, 0), 0, f32_bits(3.0))
  state.write_r(IR3Register('r', 2, 3), 1, f32_bits(7.0))
  state.write_r(IR3Register('r', 4, 0), 1, f32_bits(4.0))

  execute_instruction(state, instruction)

  assert state.read_r(IR3Register('r', 2, 3), 0) == f32_bits(5.0)
  assert state.read_r(IR3Register('r', 2, 3), 1) == f32_bits(11.0)

def test_execute_add_source_modifiers():
  dst = IR3Register('r', 0, 0)
  src0 = IR3Register('r', 1, 0)
  src1 = IR3Register('r', 2, 0)
  instruction = IR3Instruction(
    name='add.f',
    dst=dst,
    srcs=(
      IR3Source(src0, absolute=True),
      IR3Source(src1, negate=True),
    ),
  )
  state = WaveState(1, 64)

  state.write_r(src0, 0, f32_bits(-2.0))
  state.write_r(src1, 0, f32_bits(3.0))

  execute_instruction(state, instruction)

  assert state.read_r(dst, 0) == f32_bits(-1.0)

def test_decode_nop():
  expected = IR3Instruction(name='nop', dst=None, srcs=())

  assert decode_ir3(bytes.fromhex('0000000000000000')) == [expected]

def test_decode_end():
  expected = IR3Instruction(name='end', dst=None, srcs=())

  assert decode_ir3(bytes.fromhex('0000000000000003')) == [expected]

def test_execute_instruction_stream():
  instructions = decode_ir3(bytes.fromhex(
    '0000000000000000'
    '0200070002081850'
    '0200070002081850'
    '0000000000000003'
  ))
  state = WaveState(1, 64)

  state.write_r(IR3Register('r', 0, 2), 0, f32_bits(2.0))
  state.write_r(IR3Register('r', 1, 3), 0, f32_bits(3.0))

  execute_instructions(state, instructions)

  assert state.read_r(IR3Register('r', 0, 2), 0) == f32_bits(8.0)

def test_half_register_storage():
  state = WaveState(2, 64)
  register = IR3Register('hr', 2, 1)

  state.write_hr(register, 0, 0x12345)
  state.write_hr(register, 1, 0xabcd)

  assert state.read_hr(register, 0) == 0x2345
  assert state.read_hr(register, 1) == 0xabcd
