for %%e in (128 256 512) do (
    for %%l in (0.0001 0.00001) do (
        for %%n in (1 2 3) do (
            for  %%r in (1 2) do (
                for %%d in (cora citeseer pubmed) do (
                    for %%y in (2) do (
                        for %%h in (4) do (
                            start /w cmd /c python graphormer_train_multigpu.py --embedding_dim %%e --lr %%l --num_hops %%n --run_no %%r --dataset %%d --encoder_layers %%y --num_attention_heads %%h --epochs 1 
                        )
                    )
                )
            )
        )
    )
)