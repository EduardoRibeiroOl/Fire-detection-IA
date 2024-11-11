# Projeto de Detecção de Fogo em Tempo Real com Raspberry Pi e TensorFlow

## Visão Geral
Este projeto utiliza um Raspberry Pi para executar uma Inteligência Artificial de detecção de fogo em tempo real. O sistema usa uma câmera conectada ao Raspberry Pi para capturar imagens, processa os quadros utilizando TensorFlow, e exibe uma mensagem visual na tela caso seja detectado fogo. Este modelo é ideal para uso em sistemas de monitoramento e segurança, onde é necessário identificar rapidamente situações de risco.

---

## Estrutura do Projeto
- **Modelo**: `fire_detection_model.keras` (modelo de rede neural treinado para detectar fogo em imagens)
- **Script**: `IA-FULL.py` (código para captura de vídeo, processamento de imagem e classificação em tempo real)
- **Dependências**: Python 3.10 ou superior, TensorFlow, OpenCV, NumPy

---

## Pré-requisitos

### Hardware
- Raspberry Pi (recomendado modelos 3B ou superiores)
- Câmera USB ou módulo de câmera compatível com o Raspberry Pi
- Fonte de alimentação adequada para o Raspberry Pi

### Software
- Raspberry Pi OS atualizado
- Python 3.10 ou superior
