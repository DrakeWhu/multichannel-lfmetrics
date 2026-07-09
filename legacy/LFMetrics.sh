#!/bin/bash

#SBATCH --partition=novas
#SBATCH --hint=nomultithread
#SBATCH --output=LFMetrics.out
#SBATCH --error=LFMetrics.err

source ~/workspaces/cursor_workspace/Python/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/jrope/workspaces/cursor_workspace/Python/parameter_scan

INITIAL_DIR=$PWD
DB_NAME=$(basename $INITIAL_DIR).db

# ID del array -> directorio con zero-padding
printf -v directory "combination_%04d" "$SLURM_ARRAY_TASK_ID"
echo "[LFMetrics] SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID -> directory=$directory"

# Si no existe, salimos sin error (no es fallo lógico)
if [ ! -d "$directory" ]; then
  echo "[LFMetrics] SKIP: no existe $directory en $(pwd)"
  exit 0
fi

cd "$directory"
echo "[LFMetrics] PWD=$(pwd)"

LOG_FILE="LFMetrics.log"
metrics_str='[]'

# Redirigir stdout y stderr al archivo de log
exec > >(tee -a "$LOG_FILE") 2>&1

update_status() {
    local status=$1
    local retries=20
    local delay=5
    echo "Actualizando estado a $status"
    until sqlite3 $INITIAL_DIR/$DB_NAME "UPDATE simulations SET status = '$status', updated_at = datetime('now') WHERE id = $SLURM_ARRAY_TASK_ID;"
    do
        ((retries--))
        if [ $retries -lt 0 ]; then
            echo "No se pudo actualizar la base de datos después de múltiples intentos" >> "$LOG_FILE"
            return 1
        fi
        sleep $delay
    done
}

# Recibir el diccionario de métricas como argumento
metrics_str=$1

# Verificar que SLURM_ARRAY_TASK_ID esté definido
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID no está definido" >> "$LOG_FILE"
    exit 1
fi

# Ejecutar el script extract_LFMetrics.py pasando el ID de la simulación y las métricas como argumentos
python3 ~/workspaces/cursor_workspace/Python/parameter_scan/commands/extract_LFMetrics.py "$SLURM_ARRAY_TASK_ID" "$metrics_str"

# Verificar si el script se ejecutó correctamente
if [ $? -eq 0 ]; then
    update_status "analyzed"
else
    update_status "failed"
fi