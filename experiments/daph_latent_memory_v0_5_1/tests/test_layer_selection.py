from daph_latent_memory.teacher.capture import select_layer_index

def test_layer_selection_bounds():assert select_layer_index(10,.5)==4;assert select_layer_index(10,1.0)==9
