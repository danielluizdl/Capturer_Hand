"""Testes do card_classifier (imagens sintéticas) e do rubric normalizado."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.card_classifier import detect_suit
from src.gabarito_parser import HandHistory
from src.scorer import _score_hand


def make_solid(bgr, size=60):
    """Cria imagem sólida de tamanho size×size com cor BGR."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = bgr
    return img


def make_hand(board, streets, winner="Hero", pot=10.0):
    return HandHistory(
        table_id="X",
        button_seat=1,
        players={},
        hole_cards=[],
        board=board,
        streets=["preflop"] + streets,
        pot_by_street={},
        total_pot=pot,
        winner=winner,
        actions=[],
    )


# --- detect_suit ---

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


# --- rubric normalizado ---

def test_board_5cards_full_score():
    """5 cartas corretas em gabarito de 5 → 30 pts."""
    gab = make_hand(["Jd", "5s", "2h", "3d", "7s"], ["flop", "turn", "river"])
    det = make_hand(["Jd", "5s", "2h", "3d", "7s"], ["flop", "turn", "river"])
    result = _score_hand(det, gab)
    assert result["scores"]["board_cards"] == 30
    assert result["scores"]["street_sequence"] == 15

def test_board_4cards_normalized():
    """4 cartas corretas em gabarito de 4 → 30 pts (máximo proporcional)."""
    gab = make_hand(["Kd", "Ad", "9d", "8d"], ["flop", "turn"])
    det = make_hand(["Kd", "Ad", "9d", "8d"], ["flop", "turn"])
    result = _score_hand(det, gab)
    assert result["scores"]["board_cards"] == 30
    assert result["scores"]["street_sequence"] == 15

def test_board_partial():
    """3 de 4 cartas corretas → round(3/4*30)=22 pts."""
    gab = make_hand(["Kd", "Ad", "9d", "8d"], ["flop", "turn"])
    det = make_hand(["Kd", "Ad", "9d"], ["flop", "turn"])
    result = _score_hand(det, gab)
    assert result["scores"]["board_cards"] == round(3 / 4 * 30)

def test_street_sequence_2of2():
    """Gabarito sem river: detectar flop+turn → 15 pts."""
    gab = make_hand(["Kd", "Ad", "9d", "8d"], ["flop", "turn"])
    det = make_hand(["Kd", "Ad", "9d", "8d"], ["flop", "turn"])
    result = _score_hand(det, gab)
    assert result["scores"]["street_sequence"] == 15

def test_street_sequence_1of2():
    """Gabarito com flop+turn, somente flop detectado → round(1/2*15)=8 pts."""
    gab = make_hand(["Kd", "Ad", "9d", "8d"], ["flop", "turn"])
    det = make_hand(["Kd", "Ad", "9d", "8d"], ["flop"])
    result = _score_hand(det, gab)
    assert result["scores"]["street_sequence"] == round(1 / 2 * 15)

def test_board_empty_gabarito():
    """Gabarito sem board → 0 pts sem crash."""
    gab = make_hand([], [])
    det = make_hand([], [])
    result = _score_hand(det, gab)
    assert result["scores"]["board_cards"] == 0

# --- _assign_positions ---

def test_assign_positions_8max():
    """8 seats com BTN no seat 3 → seat 3=BTN, 4=SB, 5=BB."""
    from src.hh_writer_ps import _assign_positions
    seats = [1, 2, 3, 4, 5, 6, 7, 8]
    pos = _assign_positions(seats, 3)
    assert pos[3] == "BTN"
    assert pos[4] == "SB"
    assert pos[5] == "BB"

def test_assign_positions_no_button():
    """BTN fora dos seats → dict vazio."""
    from src.hh_writer_ps import _assign_positions
    pos = _assign_positions([1, 2, 3], 9)
    assert pos == {}

def test_assign_positions_wrap():
    """BTN no último seat → SB no seat 1 (wrap)."""
    from src.hh_writer_ps import _assign_positions
    seats = [1, 2, 3]
    pos = _assign_positions(seats, 3)
    assert pos[3] == "BTN"
    assert pos[1] == "SB"
    assert pos[2] == "BB"


# --- _segment_hands ---

def test_segment_hands_empty():
    from src.hand_builder import _segment_hands
    assert _segment_hands([]) == []

def test_segment_hands_single_event():
    from src.hand_builder import _segment_hands
    from src.video_pipeline import TableEvent
    ev = TableEvent(0.0, 0, 0, "TBL", "board_change", 3, None, False)
    # 1 evento com board > 0 → mantido (1 elemento, board>0)
    result = _segment_hands([ev])
    assert len(result) == 1


# --- ocr_pot tolerancia ---

def test_parse_pot_value_ocr_errors():
    from src.ocr_engine import _parse_pot_value
    # O → 0, l → 1
    assert _parse_pot_value("2O.5") == 20.5
    assert _parse_pot_value("l5") == 15.0
    assert _parse_pot_value("1OO") == 100.0  # 1OO -> 100, dentro do range
    # Sanidade: pot entre 0.5 e 500
    assert _parse_pot_value("0.1") is None  # abaixo do minimo
    assert _parse_pot_value("600") is None   # acima do maximo


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
