from test.mockgpu.qcom.emu import IR3Instruction, IR3Register, IR3Source, decode_ir3

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
