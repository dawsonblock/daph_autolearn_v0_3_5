from daph_latent_memory.benchmarks.dataset import generate_dataset

def test_dataset_has_all_splits_and_no_direct_answer_leakage():
    data=generate_dataset(100,seed=1);assert {x.split for x in data}=={"train","iid","composition","ood"}
    for ex in data:ex.state.assert_no_direct_answer_encoding(ex.answer)

def test_dataset_reproducible_in_process():
    a=[x.model_dump() for x in generate_dataset(100,seed=99)];b=[x.model_dump() for x in generate_dataset(100,seed=99)];assert a==b
