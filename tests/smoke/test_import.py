import jrystal as jr


def test_import_smoke():
  assert jr.__version__ == "0.0.2"
  assert hasattr(jr, "plot")
  assert callable(jr.calc.energy)
  assert callable(jr.calc.band)
