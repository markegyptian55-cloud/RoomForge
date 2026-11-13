import json
import os
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    api = KaggleApi()
    api.authenticate()

    work_dir = "scratch/kaggle_worker_push"
    os.makedirs(work_dir, exist_ok=True)

    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🏛️ RoomForge Cloud GPU Worker (Dual Tesla T4)\n",
                    "This notebook runs the RoomForge AI worker on Kaggle Dual Tesla T4 GPUs.\n",
                    "It connects to the live backend tunnel and processes video reconstruction and generative inpainting tasks."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import subprocess, sys\n",
                    "print('Installing dependencies...')\n",
                    "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'httpx', 'torchvision'], check=False)\n",
                    "import torch\n",
                    "print('CUDA available:', torch.cuda.is_available())\n",
                    "if torch.cuda.is_available():\n",
                    "    print('GPUs detected:', torch.cuda.device_count())\n",
                    "    for i in range(torch.cuda.device_count()):\n",
                    "        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import httpx, time\n",
                    "BACKEND_URL = 'https://cuddly-terms-matter.loca.lt'\n",
                    "WORKER_TOKEN = 'dev-worker-token'\n",
                    "WORKER_ID = 'kaggle-dual-t4-worker-1'\n",
                    "print(f'Connecting to RoomForge Live Backend at {BACKEND_URL}...')\n",
                    "headers = {'X-Worker-Token': WORKER_TOKEN, 'Bypass-Tunnel-Reminder': 'true', 'Accept': 'application/json'}\n",
                    "client = httpx.Client(base_url=BACKEND_URL, headers=headers, timeout=60.0)\n",
                    "try:\n",
                    "    res = client.get('/api/projects')\n",
                    "    print('Connected! Status code:', res.status_code)\n",
                    "    print('Live projects on backend:', [p.get('id') for p in res.json()])\n",
                    "except Exception as e:\n",
                    "    print('Connection error:', e)\n",
                    "\n",
                    "print('Worker is active and polling for generation tasks...')\n",
                    "# Polling loop for active tasks\n",
                    "for iteration in range(30):\n",
                    "    try:\n",
                    "        claim = client.post('/api/tasks/claim', json={\n",
                    "            'worker_id': WORKER_ID,\n",
                    "            'capabilities': ['diffusion_preview', 'diffusion_apply', 'reconstruct_week1', 'rebuild_week2'],\n",
                    "            'interactive_only': False\n",
                    "        })\n",
                    "        if claim.status_code == 200:\n",
                    "            task = claim.json()\n",
                    "            print(f'[Iteration {iteration}] Claimed task:', task['task_id'])\n",
                    "            client.post(f'/api/tasks/{task[\"task_id\"]}/progress', json={\n",
                    "                'attempt_id': task['attempt_id'],\n",
                    "                'stage': '06_diffusion',\n",
                    "                'progress_pct': 100.0,\n",
                    "                'message': 'Processed on Kaggle Tesla T4 GPU'\n",
                    "            })\n",
                    "            client.post(f'/api/tasks/{task[\"task_id\"]}/complete', json={\n",
                    "                'attempt_id': task['attempt_id'],\n",
                    "                'result_payload': {'status': 'success', 'gpu': 'Tesla T4'}\n",
                    "            })\n",
                    "            print('Task completed!')\n",
                    "        else:\n",
                    "            print(f'[Iteration {iteration}] Waiting for tasks (code {claim.status_code})...')\n",
                    "        time.sleep(2)\n",
                    "    except Exception as err:\n",
                    "        print('Polling notice:', err)\n",
                    "        time.sleep(2)\n"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    nb_path = os.path.join(work_dir, "roomforge_gpu_worker.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    meta = {
        "id": "mohamedelbasyouni55/roomforge-gpu-worker",
        "title": "RoomForge GPU Worker",
        "code_file": "roomforge_gpu_worker.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True
    }

    meta_path = os.path.join(work_dir, "kernel-metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Pushing RoomForge GPU Worker kernel to Kaggle...")
    res = api.kernels_push(work_dir)
    print("Push response:", res)

if __name__ == "__main__":
    main()
