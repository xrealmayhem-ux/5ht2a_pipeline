# Regla filter: REINVENT CSV -> Pharmit-ready (.csv + .smi.gz).
# Entrypoint vigente: filtro/scripts/filter1.py (ver filtro/README.md).

rule filtro__pharmit:
    input:
        csv="results/generacion/stage2_1.csv",
    output:
        csv="results/filtro/pharmit_ready.csv",
        smi="results/filtro/pharmit_ready.smi.gz",
    params:
        filtro_dir=config["paths"]["filtro_dir"],
        score_cutoff=config["filtro"]["score_cutoff"],
        max_mols=config["filtro"]["max_mols"],
        max_stereo=config["filtro"]["max_stereoisomers"],
        sa_cutoff=config["filtro"]["sa_score_cutoff"],
        smiles_col=config["filtro"]["smiles_col"],
    conda:
        "envs/filter.yaml"
    shell:
        """
        EXTRA=""
        if [ "{params.score_cutoff}" != "None" ] && [ -n "{params.score_cutoff}" ]; then
          EXTRA="$EXTRA --score-cutoff {params.score_cutoff}"
        fi
        if [ "{params.max_mols}" != "None" ] && [ -n "{params.max_mols}" ]; then
          EXTRA="$EXTRA --max-mols {params.max_mols}"
        fi
        python {params.filtro_dir}/scripts/filter1.py \
          --input {input.csv} \
          --output results/filtro/pharmit_ready \
          --smiles-col {params.smiles_col} \
          --max-stereoisomers {params.max_stereo} \
          --sa-score-cutoff {params.sa_cutoff} \
          $EXTRA
        """

rule filtro__validate_cutoffs:
    # Auditoría opcional (no participa en producción).
    output:
        done="results/filtro/cutoff_validation/DONE",
    params:
        filtro_dir=config["paths"]["filtro_dir"],
    conda:
        "envs/filter.yaml"
    shell:
        """
        python {params.filtro_dir}/scripts/validate_cutoffs.py \
          --output-dir results/filtro/cutoff_validation && touch {output.done}
        """
