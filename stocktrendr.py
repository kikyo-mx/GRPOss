import numpy as np
from scipy.optimize import leastsq


def Fun(p, x):  # 定义拟合函数形式
    a1, a2 = p
    return a1 * x + a2


def error(p, x, y):  # 拟合残差
    return Fun(p, x) - y


def find_line(p1, p2):
    a = (p1[1] - p2[1]) / (p1[0] - p2[0])
    b = p1[1] - a * p1[0]
    return (a, b)


def find_point(y_fitted, y):
    res_list = []
    for i in range(len(y)):
        res_list.append(abs(y_fitted[i] - y[i]))
    res_list = np.array(res_list)
    sx = np.argsort(res_list)[-2:].min()
    lx = np.argsort(res_list)[-2:].max()
    return sx, lx, res_list.max()


def slip(start, end, ts):
    a, b = find_line((start, ts[start]), (end, ts[end]))
    y_fit = Fun((a, b), np.linspace(start, end, end - start + 1))
    sx, lx, res_max = find_point(y_fit, ts[start:end + 1])
    return sx + start, lx + start, a, res_max


def stock_trendr(ts, win_start, win_end, st):
    final_list = []
    # for time_windows in range(windows, len(ts), windows):
    line_list = []
    x = np.array(range(len(ts)))  # 原始数据的参数
    p0 = [0.1, 1]  # 拟合的初始参数设置
    para = leastsq(error, p0, args=(x, ts))  # 进行拟合
    y_fitted = Fun(para[0], x)  # 画出拟合后的曲线
    s0, l0, _ = find_point(y_fitted, ts)
    s0 += win_start
    l0 += win_start
    line_list.append((win_start, s0))
    line_list.append((s0, l0))
    line_list.append((l0, win_end))
    new_list = line_list

    while new_list:
        new_list = []
        for i in range(len(line_list)):
            start = line_list[i][0]
            end = line_list[i][1]
            if end == 30:
                end -= 1
            time = end - start
            if time < 5:
                continue
            sx, lx, slop, res_max = slip(start, end, ts)
            # print(slop, line_list[i])
            if any([abs(slop) > 0.2, abs(slop) > st[0] and res_max < st[1]]):
            # if any([abs(slop) > 0.03]):
                final_list.append(line_list[i])
            else:
                new_list.append((start, sx))
                new_list.append((sx, lx))
                new_list.append((lx, end))
        line_list = new_list
    final_list.sort()

    # box_list = []
    # for i in range(len(final_list) - 1):
    #     if final_list[i][1] != final_list[i + 1][0]:
    #         box_list.append((final_list[i][1], final_list[i + 1][0]))

    # w_box = []
    # for windows in box_list:
    #     win_start = windows[0]
    #     win_end = windows[1]
    #     if win_end - win_start > 100 or win_end - win_start < 5:
    #         continue
    #     if win_end - win_start < 30:
    #         w_box.append(windows)
    #         continue
    #
    #     line_list = []
    #     x = np.array(range(len(ts[win_start:win_end])))  # 创建时间序列
    #     y = ts[win_start:win_end]  # 原始数据的参数
    #     p0 = [0.1, 1]  # 拟合的初始参数设置
    #     para = leastsq(error, p0, args=(x, y))  # 进行拟合
    #     y_fitted = Fun(para[0], x)  # 画出拟合后的曲线
    #     s0, l0, _ = find_point(y_fitted, y)
    #     s0 += win_start
    #     l0 += win_start
    #     line_list.append((win_start, s0))
    #     line_list.append((s0, l0))
    #     line_list.append((l0, win_end))
    #     new_list = line_list
    #
    #     while new_list:
    #         new_list = []
    #         for i in range(len(line_list)):
    #             start = line_list[i][0]
    #             end = line_list[i][1]
    #             time = end - start
    #             if time < 5:
    #                 continue
    #             sx, lx, slop, res_max = slip(start, end, ts)
    #             if any([abs(slop) < 0.05]) and time < 15:
    #                 w_box.append(line_list[i])
    #             else:
    #                 new_list.append((start, sx))
    #                 new_list.append((sx, lx))
    #                 new_list.append((lx, end))
    #         line_list = new_list
    return final_list
