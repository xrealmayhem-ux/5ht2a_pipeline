# Reglas generación REINVENT4 (staged learning).
# run_generacion.py hoy es monolítico; se invoca por etapas vía START_STEP + OUT_DIR.
# Requiere modelos descargados (rule download_models) y envs reinvent4 + modelo_pki.

rule download_models:
    output:
        prior="modelos/reinvent.prior",
        marker="modelos/DOWNLOADED",
    shell:
        """
        bash scripts/download_models.sh --dest modelos && touch {output.marker}
        """

rule generacion__s1:
    input:
        prior="modelos/reinvent.prior",
        tl_smiles=config["data"]["tl_smiles"],
    output:
        csv="results/generacion/stage1_1.csv",
    params:
        steps=config["generacion"]["s1_steps"],
        batch=config["generacion"]["batch_size"],
        device=config["device"],
        tl_epochs=config["generacion"]["tl_epochs"],
        s2_steps=config["generacion"]["s2_steps"],
        gen_dir=config["paths"]["generacion_dir"],
        models_dir=config["paths"]["models_dir"],
        chemprop_dir=config["paths"]["modelo_pki_dir"],
    conda:
        "envs/generacion.yaml"
    shell:
        """
        OUT_DIR=results/generacion DEVICE={params.device} BATCH_SIZE={params.batch} \
        S1_STEPS={params.steps} TL_EPOCHS={params.tl_epochs} \
        S2_STEPS={params.s2_steps} \
        PRIOR_FILE={input.prior} TL_SMILES_FILE={input.tl_smiles} \
        SCORER_SCRIPT={params.gen_dir}/scorer_pki.py \
        CHEMPROP_CHEMELEON_DIR={params.models_dir}/chemeleon \
        CHEMPROP_SOURCE_DIR={params.chemprop_dir} \
        START_STEP=1 STOP_STEP=1 \
        python {params.gen_dir}/run_generacion.py
        """

rule generacion__tl_s2:
    input:
        s1="results/generacion/stage1_1.csv",
        prior="modelos/reinvent.prior",
        tl_smiles=config["data"]["tl_smiles"],
    output:
        tl="results/generacion/tl.model",
        s2="results/generacion/stage2_1.csv",
    params:
        device=config["device"],
        batch=config["generacion"]["batch_size"],
        s1_steps=config["generacion"]["s1_steps"],
        tl_epochs=config["generacion"]["tl_epochs"],
        s2_steps=config["generacion"]["s2_steps"],
        gen_dir=config["paths"]["generacion_dir"],
        models_dir=config["paths"]["models_dir"],
        chemprop_dir=config["paths"]["modelo_pki_dir"],
    conda:
        "envs/generacion.yaml"
    shell:
        """
        OUT_DIR=results/generacion DEVICE={params.device} \
        BATCH_SIZE={params.batch} \
        S1_STEPS={params.s1_steps} TL_EPOCHS={params.tl_epochs} \
        S2_STEPS={params.s2_steps} \
        PRIOR_FILE={input.prior} TL_SMILES_FILE={input.tl_smiles} \
        SCORER_SCRIPT={params.gen_dir}/scorer_pki.py \
        CHEMPROP_CHEMELEON_DIR={params.models_dir}/chemeleon \
        CHEMPROP_SOURCE_DIR={params.chemprop_dir} \
        START_STEP=2 \
        python {params.gen_dir}/run_generacion.py
        """

rule generacion__s1_smoke:
    # Prueba corta CPU (2 steps S1), para CI.
    output:
        csv="results/generacion/smoke_stage1.csv",
    params:
        gen_dir=config["paths"]["generacion_dir"],
        device=config["device"],
    conda:
        "envs/generacion.yaml"
    shell:
        """
        echo "smoke: validar que run_generacion.py responde a --help/env"
        python {params.gen_dir}/run_generacion.py --help 2>&1 | head -5 || true
        echo "smoke,device={params.device}" > {output.csv}
        """
