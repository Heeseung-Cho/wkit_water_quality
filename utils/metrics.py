import numpy as np


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    eps = 1e-3
    return np.mean(np.abs((pred - true) / np.maximum(np.abs(true), eps)))


def MSPE(pred, true):
    eps = 1e-3
    return np.mean(np.square((pred - true) / np.maximum(np.abs(true), eps)))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
