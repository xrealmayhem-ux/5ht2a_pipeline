# Reglas Chemprop: prepare_pki -> train_pki (ensemble 5 seeds).
# No tocan los checkpoints productivos salvo que se pida train_full.

rule chemprop__prepare:
    input:
        gpcrdb=config["data"]["pki_raw_gpcrdb"],
        chembl=config["data"]["pki_raw_chembl"],
    output:
        curated="results/chemprop/pki_curated.csv",
    params:
        chemprop_dir=config["paths"]["chemprop_dir"],
    conda:
        "envs/chemprop.yaml"
    shell:
        """
        python {params.chemprop_dir}/prepare_pki.py \
          --out {output.curated}
        """

rule chemprop__train_seed:
    input:
        curated="results/chemprop/pki_curated.csv",
    output:
        done="results/chemprop/checkpoints/seed_{seed}/DONE",
    params:
        seed="{seed}",
        epochs=config["chemprop"]["max_epochs"],
        chemprop_dir=config["paths"]["chemprop_dir"],
    conda:
        "envs/chemprop.yaml"
    shell:
        """
        python {params.chemprop_dir}/train_pki.py \
          --seed {params.seed} \
          --max-epochs {params.epochs} \
          --out-dir results/chemprop/checkpoints/seed_{params.seed} && \
        touch {output.done}
        """

rule chemprop__train_full:
    input:
        expand("results/chemprop/checkpoints/seed_{s}/DONE", s=SEEDS),

rule chemprop__smoke:
    # Smoke test: no toca split persistido ni checkpoints productivos.
    output:
        done="results/chemprop/smoke/DONE",
    params:
        chemprop_dir=config["paths"]["chemprop_dir"],
        limit=config["chemprop"]["smoke_limit"],
        epochs=config["chemprop"]["smoke_epochs"],
    conda:
        "envs/chemprop.yaml"
    shell:
        """
        python {params.chemprop_dir}/train_pki.py --seed 42 \
          --limit {params.limit} \
          --max-epochs {params.epochs} \
          --no-warmstart \
          --out-dir results/chemprop/smoke && touch {output.done}
        """
