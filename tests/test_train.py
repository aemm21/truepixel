import math

from train import calcular_auc, calcular_metricas, dividir_estratificado


def test_split_estratificado_preserva_clases():
    etiquetas = [0] * 70 + [1] * 30
    train_idx, val_idx, test_idx = dividir_estratificado(
        etiquetas, val_split=0.2, test_split=0.15, seed=42
    )

    assert len(train_idx) + len(val_idx) + len(test_idx) == 100
    assert set(train_idx).isdisjoint(val_idx)
    assert set(train_idx).isdisjoint(test_idx)
    assert set(val_idx).isdisjoint(test_idx)

    for indices in (train_idx, val_idx, test_idx):
        assert any(etiquetas[i] == 0 for i in indices)
        assert any(etiquetas[i] == 1 for i in indices)


def test_auc_perfecta_e_invertida():
    y_true = [0, 0, 1, 1]
    assert calcular_auc(y_true, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert calcular_auc(y_true, [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_auc_una_sola_clase_es_nan():
    assert math.isnan(calcular_auc([0, 0, 0], [0.1, 0.2, 0.3]))


def test_metricas_basicas():
    metricas = calcular_metricas(
        y_true=[0, 1, 1, 0],
        y_prob=[0.9, 0.8, 0.2, 0.1],
        umbral=0.5,
    )
    assert metricas["tp"] == 1
    assert metricas["tn"] == 1
    assert metricas["fp"] == 1
    assert metricas["fn"] == 1
    assert metricas["accuracy"] == 0.5
    assert metricas["precision"] == 0.5
    assert metricas["recall"] == 0.5
    assert metricas["f1"] == 0.5


def test_metricas_perfectas():
    metricas = calcular_metricas(
        y_true=[0, 0, 1, 1],
        y_prob=[0.1, 0.4, 0.6, 0.9],
        umbral=0.5,
    )
    assert metricas["accuracy"] == 1.0
    assert metricas["precision"] == 1.0
    assert metricas["recall"] == 1.0
    assert metricas["f1"] == 1.0
    assert metricas["auc"] == 1.0
