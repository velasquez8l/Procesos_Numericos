import json

info_metodos = {
    "1": {
        "biseccion": {
            "x0": None,
            "x1": None,
            "TOL": None,
            "N_max": None,
        }
    },
    "2": {
        "regla_falsa": {
            "x0": None,
            "x1": None,
            "TOL": None,
            "N_max": None,
        }
    },
    "3": {
        "secante": {
            "x0": None,
            "x1": None,
            "TOL": None,
            "N_max": None,
        }
    },
    "4": {
        "newton": {
            "x0": None,
            "TOL": None,
            "N_max": None,
        }
    },
    "5": {
        "punto_fijo": {
            "x0": None,
            "TOL": None,
            "N_max": None,
        }
    },
    "6": {"busqueda_incremental": {"x0": None, "delta": None, "N_max": None}},
    "7": {
        "newton_ceros_multiples": {
            "x0": None,
            "TOL": None,
            "N_max": None,
        }
    },
}
with open("datos.json", "w") as f:
    json.dump(info_metodos, f)
