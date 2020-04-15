from getalp.wsd.predicter_online import Predicter

predicter = Predicter()
predicter.training_root_path = "/tf/model"
predicter.ensemble_weights_path = ["/tf/model/model_weights_wsd0"]
predicter.clear_text = True
predicter.batch_size = 1
predicter.disambiguate = True
predicter.beam_size = 1
predicter.output_all_features = false