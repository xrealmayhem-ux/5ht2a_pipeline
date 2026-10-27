# Reglas modelo_pki: entrenamiento del modelo CheMeleon (fine-tune pKi).

rule modelo_pki__train_seed:
    input:
        curated="results/curacion/pki_curated.csv",
    output:
        done="results/modelo_pki/checkpoints/seed_{seed}/DONE",
    params:
        seed="{seed}",
        epochs=config["modelo_pki"]["max_epochs"],
        modelo_pki_dir=config["paths"]["modelo_pki_dir"],
    conda:
        "envs/modelo_pki.yaml"
    shell:
        """
        python {params.modelo_pki_dir}/train_pki.py \
          --seed {params.seed} \
          --max-epochs {params.epochs} \
          --out-dir results/modelo_pki/checkpoints/seed_{params.seed} && \
        touch {output.done}
        """

rule modelo_pki__train_full:
    input:
        expand("results/modelo_pki/checkpoints/seed_{s}/DONE", s=SEEDS),
    params:
        seeds=SEEDS,
        epochs=config["modelo_pki"]["max_epochs"],
        modelo_pki_dir=config["paths"]["modelo_pki_dir"],
    conda:
        "envs/modelo_pki.yaml"
    shell:
        """
        echo "modelo_pki__train_full: corrida multi-seed completada (sin pasos shell pendientes)"
        """

rule modelo_pki__smoke:
    # Smoke test: no toca split persistido ni checkpoints productivos.
    output:
        done="results/modelo_pki/smoke/DONE",
    params:
        modelo_pki_dir=config["paths"]["modelo_pki_dir"],
        limit=config["modelo_pki"]["smoke_limit"],
        epochs=config["modelo_pki"]["smoke_epochs"],
    conda:
        "envs/modelo_pki.yaml"
    shell:
        """
        python {params.modelo_pki_dir}/train_pki.py --seed 42 \
          --limit {params.limit} \
          --max-epochs {params.epochs} \
          --no-warmstart \
          --out-dir results/modelo_pki/smoke && touch {output.done}
        """
