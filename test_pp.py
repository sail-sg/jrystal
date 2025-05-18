from jrystal.pseudopotential.load import parse_upf


def main():
  # pp_dict = parse_upf('/home/aiops/zhaojx/jrystal/pseudopotential/normconserving/C.pz-vbc.UPF')
  # pp_dict = parse_upf("/home/aiops/zhaojx/jrystal/pseudopotential/C.pbe-n-rrkjus_psl.1.0.1.UPF")
  pp_dict = parse_upf("/home/aiops/zhaojx/jrystal/pseudopotential/C.pbe-n-kjpaw_psl.1.0.0.UPF")
  breakpoint()

if __name__ == "__main__":
  main()
