from getalp.wsd.predicter_online import Predicter

predicter = Predicter()
predicter.training_root_path = "/tf/model"
predicter.ensemble_weights_path = ["/tf/model/model_weights_wsd0"]
predicter.clear_text = True
predicter.batch_size = 1
predicter.disambiguate = True
predicter.beam_size = 1
predicter.output_all_features = false

lines = ["Andre Agassi Andre Kirk Agassi -LRB- ; born April 29 , 1970 -RRB- is an American retired professional tennis player and former world No. 1 whose career spanned from the late 1980s to the early 2000s . ", "0 0 0 1066 0 0 0 2879;285;10217;1615;220;3057;425;5333;1258;2032;3753;1789;405 18 0 0 0 0 1429;1757;67;2;5770;6;179;38;426;1929;3178;792;98 0 222;587 6067 2028;4935;10233;7857;6284 3076 615;367;409;1383 0 4390;4708;525;2929 1057;73;653;739;1281;694;233;495 254 0 0 2606;47 3751 0 0 12721;3260;14966;14850;2929;6098;7546 0 0 0 1811;4133;14885;10988;4708 0 0 "]
