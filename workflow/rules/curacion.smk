# Regla curacion: curación de mediciones Ki/pKi (GPCRdb + ChEMBL) -> curated CSV.
# Etapa 1 del flujo: produce el input del entrenamiento del modelo (modelo_pki).

rule curacion:
    input:
        gpcrdb=config["data"]["pki_raw_gpcrdb"],
        chembl=config["data"]["pki_raw_chembl"],
    output:
        curated="results/curacion/pki_curated.csv",
    params:
        curacion_dir=config["paths"]["curacion_dir"],
    conda:
        "envs/curacion.yaml"
    shell:
        """
        python {params.curacion_dir}/prepare_pki.py \
          --out {output.curated}
        """
