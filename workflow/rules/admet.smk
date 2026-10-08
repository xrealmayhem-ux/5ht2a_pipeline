# Reglas ADMET: predict_admet -> filter_safety (regla dura hERG/AMES/DILI/Carcinogens>=0.50).
# Entrada recomendada = CSV final de filter/ (ver admet_ai_filter/README.md).

rule admet__predict:
    input:
        csv="results/filter/pharmit_ready.csv",
    output:
        pred="results/admet/admet_predictions.csv",
    params:
        admet_dir=config["paths"]["admet_dir"],
        smiles_col=config["admet"]["smiles_col"],
    conda:
        "envs/admet.yaml"
    shell:
        """
        python {params.admet_dir}/scripts/predict_admet.py \
          --input {input.csv} \
          --output {output.pred} \
          --smiles-col {params.smiles_col}
        """

rule admet__safety:
    input:
        pred="results/admet/admet_predictions.csv",
    output:
        passed="results/admet/admet_safety_passed.csv",
    params:
        admet_dir=config["paths"]["admet_dir"],
        cutoff=config["admet"]["risk_cutoff"],
    conda:
        "envs/admet.yaml"
    shell:
        """
        python {params.admet_dir}/scripts/filter_safety.py \
          --input {input.pred} \
          --output {output.passed} \
          --risk-cutoff {params.cutoff}
        """
