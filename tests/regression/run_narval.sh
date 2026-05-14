#!/bin/bash
#SBATCH --job-name=starships_regression
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/regression_%j.out
#SBATCH --error=logs/regression_%j.err

# =============================================================================
# Script pour rouler les tests de régression sur Narval (Calcul Québec)
#
# Usage :
#   Première fois (génération des golden outputs) :
#     sbatch --export=MODE=generate run_narval.sh
#
#   Après chaque modification (validation) :
#     sbatch --export=MODE=validate run_narval.sh
#
# Variables à ajuster :
#   STARSHIPS_DIR  : chemin vers le dossier starships/
#   VENV           : chemin vers l'environnement virtuel
#   CONFIG         : chemin vers config.yaml
# =============================================================================

STARSHIPS_DIR="${STARSHIPS_DIR:-/path/to/starships}"
VENV="${VENV:-/path/to/venv/starships_39}"
CONFIG="${CONFIG:-${STARSHIPS_DIR}/tests/regression/config.yaml}"
MODE="${MODE:-validate}"

# Charger les modules Narval
module load StdEnv/2020
module load gcc python/3.9 mpi4py hdf5 netcdf

source "${VENV}/bin/activate"

cd "${STARSHIPS_DIR}"
mkdir -p logs

echo "============================================"
echo " STARSHIPS — Tests de régression"
echo " Mode    : ${MODE}"
echo " Config  : ${CONFIG}"
echo " Date    : $(date)"
echo "============================================"

if [ "${MODE}" = "generate" ]; then
    echo "Génération des golden outputs..."
    python tests/regression/generate_golden.py --config "${CONFIG}"

elif [ "${MODE}" = "validate" ]; then
    echo "Validation (comparaison avec golden outputs)..."
    python -m pytest tests/regression/ -v -m regression \
        --tb=short \
        --no-header \
        2>&1

    # Rouler aussi les tests unitaires pour être complet
    echo ""
    echo "Tests unitaires (locaux) :"
    python -m pytest tests/unit/ -v --tb=short --no-header 2>&1

else
    echo "ERREUR: MODE inconnu '${MODE}'. Utiliser 'generate' ou 'validate'."
    exit 1
fi

echo "============================================"
echo " Terminé : $(date)"
echo "============================================"
