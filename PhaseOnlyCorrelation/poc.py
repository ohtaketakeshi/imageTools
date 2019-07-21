if __debug__:
    print("import modules")
from typing import List
import argparse
import cv2 as cv
import numpy as np
from scipy import signal, fftpack
from scipy.optimize import leastsq
from PIL import Image
import pathlib
if __debug__:
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

def imread(filename, flags=cv.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv.imdecode(n, flags)
        if img is None:
            pil = Image.open(filename)
            img = np.asarray(pil)
            if img.ndim == 2:
                pass
            elif img.shape[2] == 3:
                img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            elif img.shape[2] == 4:
                img = cv.cvtColor(img, cv.COLOR_RGBA2BGRA)
        return img
    except Exception as e:
        print(e)
        return None

def imwrite(filename, img, params=None):
    try:
        ext = pathlib.Path(filename).suffix
        result, n = cv.imencode(ext, img, params)

        if result:
            with open(filename, mode="wb") as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False

def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def warpImage(img, angle, scale, dy, dx):
    center = np.array(img.shape)[0:2]/2
    rotMat = cv.getRotationMatrix2D((center[1], center[0]), angle, scale)
    rotMat = np.vstack([rotMat, [0,0,1]])
    shiftMat = np.float32([
        [1,0,dx],
        [0,1,dy],
        [0,0,1]
    ])
    M = np.matmul(shiftMat, rotMat)
    return cv.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv.INTER_LANCZOS4)

def logpolar(src, center, M):
    srcshape = src.shape[0:2]
    dstshape = (4096, 1024)
    return cv.warpPolar(
        src, dstshape, center, M,
        flags=cv.WARP_POLAR_LOG+cv.INTER_CUBIC+cv.WARP_FILL_OUTLIERS
        )

"""
def poc_func(n1, n2, alpha, delta1, delta2, N1, N2):
    d1 = (n1 + delta1)
    d2 = (n2 + delta2)
    with np.errstate(divide='ignore', invalid='ignore'):
        coef1 = np.where( d1==0, N1 ,np.sin(np.pi*d1) / np.sin(np.pi/N1*d1) )
        coef2 = np.where( d2==0, N2 ,np.sin(np.pi*d2) / np.sin(np.pi/N2*d2) )
    return alpha / (N1*N2) * coef1 * coef2
"""
def poc_func_sinc():
    return lambda n1, n2, alpha, delta1, delta2: alpha * np.sinc(n1 + delta1) * np.sinc(n2 + delta2)

def poc_func_gaussian(sigma):
    return lambda n1, n2, alpha, delta1, delta2: alpha/(2*np.pi*sigma**2) * np.exp(-1/2*((n1+delta1)/sigma)**2) * np.exp(-1/2*((n2+delta2)/sigma)**2)

def poc(f, g, fitting_shape = (9,9), with_LPF=True, debug_info=None):
    N1, N2 = f.shape
    win_y = signal.windows.hann(N1)
    win_x = signal.windows.hann(N2)
    win = win_y.reshape((-1,1)) * win_x
    f_win = f * win
    g_win = g * win

    F = fftpack.fft2(f_win)
    G = fftpack.fft2(g_win)
    G_ = np.conj(G)
    R = F * G_
    R = R / np.abs(R)

    if with_LPF:
        #R = fftpack.fftshift(R)
        #R_center = np.floor(np.array(R.shape)/2).astype(int)
        sigma = 0.71
        LPF_y = np.exp(-2*(np.pi*sigma*fftpack.fftfreq(N1))**2)
        LPF_x = np.exp(-2*(np.pi*sigma*fftpack.fftfreq(N2))**2)
        LPF = LPF_y.reshape((-1,1)) * LPF_x
        R = R * LPF
        poc_func = poc_func_gaussian(sigma)
    else:
        poc_func = poc_func_sinc()


    r = fftpack.fftshift(np.real(fftpack.ifft2(R)))
    r_center = np.floor(np.array(r.shape)/2).astype(int)

    if __debug__:
        F_abs = fftpack.fftshift(np.log(np.abs(F)+1E-30))
        G_abs = fftpack.fftshift(np.log(np.abs(G)+1E-30))
        R_abs = fftpack.fftshift(np.log(np.abs(R)+1E-30))
        imwrite("debug/poc_{}_f_.png".format(debug_info), normalize(f)*255)
        imwrite("debug/poc_{}_g_.png".format(debug_info), normalize(g)*255)
        imwrite("debug/poc_{}_f_win.png".format(debug_info), normalize(f_win)*255)
        imwrite("debug/poc_{}_g_win.png".format(debug_info), normalize(g_win)*255)
        imwrite("debug/poc_{}_F.png".format(debug_info), normalize(F_abs) * 255)
        imwrite("debug/poc_{}_G.png".format(debug_info), normalize(G_abs) * 255)
        imwrite("debug/poc_{}_R.png".format(debug_info), normalize(R_abs) * 255)
        imwrite("debug/poc_{}_r_.png".format(debug_info), normalize(r) * 255)

    max_pos = np.argmax(r)
    peak = (max_pos // N2, max_pos % N2)
    if __debug__:
        print(max_pos)
        print("peak:", peak)
        peak_value = r[peak[0], peak[1]]
        print("peak_value:", peak_value)

    mf = np.floor(np.array(fitting_shape)/2+1E-30).astype(int)
    fitting_area = r[
        peak[0] - mf[0]: peak[0] + mf[1] + 1,
        peak[1] - mf[1]: peak[1] + mf[1] + 1
        ]
    p0 = [1, 0.000, 0.000] # initial value
    grid_y, grid_x = np.mgrid[-mf[0]:mf[0]+1, -mf[1]:mf[1]+1]
    error_func = lambda p: np.ravel(poc_func(grid_y, grid_x, p[0], p[1], p[2]) - fitting_area)
    plsq = leastsq(error_func, p0)

    alpha = plsq[0][0]
    delta1 = plsq[0][1]
    delta2 = plsq[0][2]
    delta_y = peak[0] - delta1 - r_center[0]
    delta_x = peak[1] - delta2 - r_center[1]

    if __debug__:
        print("plsq:", plsq)
        r_img = (normalize(r) * 255).astype(np.uint8)
        r_marked = cv.drawMarker(cv.cvtColor(r_img, cv.COLOR_GRAY2BGR), (peak[1], peak[0]), (0,0,255), cv.MARKER_CROSS)
        imwrite("debug/poc_{}_r_marked.png".format(debug_info), r_marked)

        # 3D plot
        fig:plt.Figure = plt.figure(figsize=(12,8))
        ax:Axes3D = fig.add_subplot(1,1,1, projection="3d")
        lin_y = np.linspace(-mf[0],mf[0],101)
        lin_x = np.linspace(-mf[1],mf[1],101)
        lin_grid_y, lin_grid_x = np.meshgrid(lin_y, lin_x)
        ax.plot_wireframe(lin_grid_x, lin_grid_y, poc_func(lin_grid_y, lin_grid_x, plsq[0][0], plsq[0][1], plsq[0][2]), color=(0.5,0.5,0.8), rstride=5, cstride=5)
        ax.scatter(grid_x, grid_y, fitting_area, c=((1,0,0),))
        fig.savefig("debug/poc_{}_fitting_3D.png".format(debug_info))

        # 2D plot
        fig, axs = plt.subplots(1,2,figsize=(12,8))
        fig: plt.Figure
        axs: List[plt.Axes]
        lin_y = np.linspace(-mf[0],mf[0],100)
        lin_x = np.linspace(-mf[1],mf[1],100)
        axs[0].plot(lin_x, poc_func(0, lin_x, plsq[0][0], plsq[0][1], plsq[0][2]), label="poc_func")
        axs[0].scatter(grid_x[mf[0],:], fitting_area[mf[0],:], label="fitting_area")
        axs[0].axvline(0)
        axs[0].axvline(-plsq[0][2])
        axs[0].legend()
        axs[1].plot(lin_y, poc_func(lin_y, 0, plsq[0][0], plsq[0][1], plsq[0][2]), label="poc_func")
        axs[1].scatter(grid_y[:,mf[1]], fitting_area[:,mf[1]], label="fitting_area")
        axs[1].axvline(0)
        axs[1].axvline(-plsq[0][1])
        axs[1].legend()
        fig.savefig("debug/poc_{}_fitting.png".format(debug_info))
    
    if __debug__:
        print("alpha:", alpha)

    return delta_y, delta_x

def ripoc(f, g):
    if f.shape != g.shape:
        return None
    h, w = f.shape
    win_y = signal.windows.hann(h)
    win_x = signal.windows.hann(w)
    win = win_y.reshape((-1,1)) * win_x
    f_win = f * win
    g_win = g * win

    F = fftpack.fft2(f_win)
    G = fftpack.fft2(g_win)
    F = fftpack.fftshift(np.log(np.abs(F)+1E-30))
    G = fftpack.fftshift(np.log(np.abs(G)+1E-30))
    F_center = (F.shape[0] / 2, F.shape[1] / 2)
    G_center = (G.shape[0] / 2, G.shape[1] / 2)
    F_M = np.sqrt((F.shape[0]/2)**2 + (F.shape[1]/2)**2)
    G_M = np.sqrt((G.shape[0]/2)**2 + (G.shape[1]/2)**2)

    if __debug__:
        print("F_M:", F_M)
        print("G_M:", G_M)
        imwrite("debug/flp_.png", logpolar(f, (F_center[1], F_center[0]), F_M))
    
    FLP = logpolar(F, (F_center[1], F_center[0]), F_M)
    LP_h, LP_w = FLP.shape
    GLP = logpolar(G, (G_center[1], G_center[0]), G_M)
    if __debug__:
        imwrite("debug/f.png", f)
        imwrite("debug/g.png", g)
        imwrite("debug/f_win.png", f_win)
        imwrite("debug/g_win.png", g_win)
        imwrite("debug/F.png", normalize(F) * 255)
        imwrite("debug/G.png", normalize(G) * 255)
        imwrite("debug/FLP.png", normalize(FLP) * 255)
        imwrite("debug/GLP.png", normalize(GLP) * 255)
    
    delta = poc(FLP, GLP, debug_info="rot_scale")
    if __debug__:
        GLP_matched = warpImage(GLP, 0, 1, delta[0], delta[1])
        imwrite("debug/FLP_GLP_matched.png", normalize(GLP_matched) * 255)
    angle = -delta[0] / LP_h * 360
    scale = 1/(np.exp(delta[1] / LP_w * np.log(F_M)))

    if __debug__:
        print("delta:", delta)
        print("angle:", angle)
        print("scale:", scale)

    center = np.array(g.shape)/2
    rotMat = cv.getRotationMatrix2D((center[1], center[0]), angle, scale)
    g_dash = cv.warpAffine(g, rotMat, (w, h), flags=cv.INTER_LANCZOS4)

    dy, dx = poc(f, g_dash, debug_info="shift")
    if __debug__:
        transMat = np.float32([
            [1,0,dx],
            [0,1,dy]
        ])
        g_matched = cv.warpAffine(g_dash, transMat, (w, h), flags=cv.INTER_LANCZOS4)
        imwrite("debug/g_matched.png", g_matched)
    return angle, scale, dy, dx

def padding_image(src, dstshape, pos=None):
    srcshape = src.shape
    pad_h = dstshape[0] - srcshape[0]
    pad_w = dstshape[1] - srcshape[1]
    if pad_h < 0:
        pad_h = 0
    if pad_w < 0:
        pad_w = 0
    if pos is None:
        pos = (pad_h // 2, pad_w // 2)
    if src.ndim == 2:
        pad_width = (
            (pos[0], pad_h - pos[0]),
            (pos[1], pad_w - pos[1])
        )
    elif src.ndim == 3:
        pad_width = (
            (pos[0], pad_h - pos[0]),
            (pos[1], pad_w - pos[1]),
            (0,0)
        )
    return np.pad(src, pad_width, mode='constant', constant_values=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("target")
    parser.add_argument("matched")
    args = parser.parse_args()
    if __debug__:
        pathlib.Path("debug").mkdir(exist_ok=True)
    f = imread(args.reference, cv.IMREAD_COLOR)
    g = imread(args.target, cv.IMREAD_COLOR)
    g = padding_image(g, f.shape)
    f = padding_image(f, g.shape)
    f_gray = cv.cvtColor(f, cv.COLOR_BGR2GRAY)
    g_gray = cv.cvtColor(g, cv.COLOR_BGR2GRAY)
    angle, scale, dy, dx = ripoc(f_gray, g_gray)
    g_matched = warpImage(g, angle, scale, dy, dx)
    imwrite(args.matched, g_matched)
    print(dy, dx, angle, scale)
    if __debug__:
        imwrite("debug/f_pad.png", f)
        imwrite("debug/g_pad.png", g)
