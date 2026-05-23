"""Testes básicos do card_classifier usando imagens sintéticas."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.card_classifier import detect_suit

def make_solid(bgr, size=60):
    """Cria imagem sólida de tamanho size×size com cor BGR."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = bgr
    return img

def test_clubs_green():
    img = make_solid((0, 180, 0))   # BGR verde vivo
    assert detect_suit(img) == "c"

def test_diamonds_blue():
    img = make_solid((200, 50, 0))  # BGR azul vivo
    assert detect_suit(img) == "d"

def test_hearts_red():
    img = make_solid((0, 0, 200))   # BGR vermelho vivo
    assert detect_suit(img) == "h"

def test_spades_dark():
    img = make_solid((30, 30, 30))  # BGR quase preto
    assert detect_suit(img) == "s"

def test_spades_dark2():
    img = make_solid((60, 60, 60))  # BGR cinza escuro
    assert detect_suit(img) == "s"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
