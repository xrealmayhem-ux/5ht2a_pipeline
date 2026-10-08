# Regla filter: REINVENT CSV -> Pharmit-ready (.csv + .smi.gz).
# Entrypoint vigente: scripts/filter1.py (ver filter/README.md).

rule filter__pharmit:
    input:
        csv="results/generacion/stage2_1.csv",
    output:
        csv="results/filter/pharmit_ready.csv",
        smi="results/filter/pharmit_ready.smi.gz",
    params:
        filter_dir=config["paths"]["filter_dir"],
        score_cutoff=config["filter"]["score_cutoff"],
        max_mols=config["filter"]["max_mols"],
        max_stereo=config["filter"]["max_stereoisomers"],
        sa_cutoff=config["filter"]["sa_score_cutoff"],
        smiles_col=config["filter"]["smiles_col"],
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
        python {params.filter_dir}/scripts/filter1.py \
          --input {input.csv} \
          --output results/filter/pharmit_ready \
          --smiles-col {params.smiles_col} \
          --max-stereoisomers {params.max_stereo} \
          --sa-score-cutoff {params.sa_cutoff} \
          $EXTRA
        """

rule filter__validate_cutoffs:
    # Auditoría opcional (no participa en producción).
    output:
        done="results/filter/cutoff_validation/DONE",
    params:
        filter_dir=config["paths"]["filter_dir"],
    conda:
        "envs/filter.yaml"
    shell:
        """
        python {params.filter_dir}/scripts/validate_cutoffs.py \
          --output-dir results/filter/cutoff_validation && touch {output.done}
        """
