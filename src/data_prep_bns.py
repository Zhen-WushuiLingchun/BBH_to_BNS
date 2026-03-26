"""
BNS 数据生成

本模块保持原始数据管线的函数接口与输出形状不变，但在 BNS 分支中
使用带潮汐效应的波形模型。状态方程固定为 APR4_EPP，潮汐参数通过
LALSimulation 的 TOV solver 计算。

这里采用严格模式：如果当前 LALSuite 环境不支持所需的潮汐近似模型
或相关参数接口，则直接报错，不做静默回退。
"""

from __future__ import division

import lal
import lalsimulation
from lal.antenna import AntennaResponse
from lal import MSUN_SI, C_SI, G_SI

import sys
import time
import numpy as np
from scipy.optimize import brentq
from scipy import interpolate


if sys.version_info >= (3, 0):
    xrange = range

safe = 2

# ----------------------------------------------------------------
# 固定 EOS 配置
# ----------------------------------------------------------------
BNS_EOS_NAME = 'APR4_EPP'
BNS_MASS_MIN = 1.0   # Msun
BNS_MASS_MAX = 2.0   # Msun, 低于 APR4_EPP 的 M_max = 2.159

# ----------------------------------------------------------------
# EOS family 缓存（TOV 积分只做一次）
# ----------------------------------------------------------------
_eos_family_cache = {}


def _get_eos_family(eos_name):
    """获取或创建指定 EOS 的 neutron star family（含 TOV 序列）。"""
    if eos_name not in _eos_family_cache:
        eos = lalsimulation.SimNeutronStarEOSByName(eos_name)
        fam = lalsimulation.CreateSimNeutronStarFamily(eos)
        _eos_family_cache[eos_name] = fam
    return _eos_family_cache[eos_name]


def compute_lambda_from_eos(m_solar, eos_name=BNS_EOS_NAME):
    """
    根据给定 EOS，利用 LALSimulation 的 TOV solver 计算质量为 m_solar
    的中子星无量纲潮汐形变参数 Lambda。
    """
    fam = _get_eos_family(eos_name)
    m_SI = float(m_solar) * lal.MSUN_SI
    r = lalsimulation.SimNeutronStarRadius(m_SI, fam)
    k2 = lalsimulation.SimNeutronStarLoveNumberK2(m_SI, fam)
    C = lal.G_SI * m_SI / (r * lal.C_SI ** 2)
    Lambda = (2.0 / 3.0) * k2 * C ** (-5)
    return float(Lambda)


# ================================================================
# 参数容器 — 扩展了 eos_name, lambda1, lambda2
# ================================================================
class bnsparams:
    """
    BNS 参数容器。

    旧字段 (mc, M, eta, m1, m2, ra, dec, iota, phi, psi, idx, fmin,
    snr, SNR) 完全保留，下游代码不受影响。
    新增: eos_name, lambda1, lambda2，用于诊断和后验分析。
    """
    def __init__(self, mc, M, eta, m1, m2, ra, dec, iota, phi, psi,
                 idx, fmin, snr, SNR,
                 eos_name=None, lambda1=None, lambda2=None):
        self.mc = mc
        self.M = M
        self.eta = eta
        self.m1 = m1
        self.m2 = m2
        self.ra = ra
        self.dec = dec
        self.iota = iota
        self.phi = phi
        self.psi = psi
        self.idx = idx
        self.fmin = fmin
        self.snr = snr
        self.SNR = SNR
        # 新增字段
        self.eos_name = eos_name
        self.lambda1 = lambda1
        self.lambda2 = lambda2


# ================================================================
# 通用工具函数 — 与 BBH 版完全一致
# ================================================================
def tukey(M, alpha=0.5):
    n = np.arange(0, M)
    width = int(np.floor(alpha * (M - 1) / 2.0))
    n1 = n[0:width + 1]
    n2 = n[width + 1:M - width - 1]
    n3 = n[M - width - 1:]
    w1 = 0.5 * (1 + np.cos(np.pi * (-1 + 2.0 * n1 / alpha / (M - 1))))
    w2 = np.ones(n2.shape)
    w3 = 0.5 * (1 + np.cos(np.pi * (-2.0 / alpha + 1 + 2.0 * n3 / alpha / (M - 1))))
    w = np.concatenate((w1, w2, w3))
    return np.array(w[:M])


def convert_beta(beta, fs, T_obs):
    newbeta = np.array([(beta[0] + 0.5 * safe - 0.5),
                        (beta[1] + 0.5 * safe - 0.5)]) / safe
    low_idx = int(T_obs * fs * newbeta[0])
    high_idx = int(T_obs * fs * newbeta[1])
    return low_idx, high_idx


def gen_noise(fs, T_obs, psd):
    N = T_obs * fs
    Nf = N // 2 + 1
    df = 1 / T_obs
    amp = np.sqrt(0.25 * T_obs * psd)
    idx = np.argwhere(psd == 0.0)
    amp[idx] = 0.0
    re = amp * np.random.normal(0, 1, Nf)
    im = amp * np.random.normal(0, 1, Nf)
    re[0] = 0.0
    im[0] = 0.0
    x = N * np.fft.irfft(re + 1j * im) * df
    return x


def gen_psd(fs, T_obs, op='AdvDesign', det='H1'):
    N = T_obs * fs
    df = 1 / T_obs
    psd = lal.CreateREAL8FrequencySeries(None, lal.LIGOTimeGPS(0), 0.0, df,
                                         lal.HertzUnit, N // 2 + 1)
    if det in ('H1', 'L1'):
        lalsimulation.SimNoisePSDAdVDesignSensitivityP1200087(psd, 10.0)
    else:
        raise ValueError('Unknown detector: {}'.format(det))
    return psd


def get_snr(data, T_obs, fs, psd, fmin):
    N = T_obs * fs
    df = 1.0 / T_obs
    dt = 1.0 / fs
    fidx = int(fmin / df)
    win = tukey(N, alpha=1.0 / 8.0)
    idx = np.argwhere(psd > 0.0)
    invpsd = np.zeros(psd.size)
    invpsd[idx] = 1.0 / psd[idx]
    xf = np.fft.rfft(data * win) * dt
    SNRsq = 4.0 * np.sum((np.abs(xf[fidx:]) ** 2) * invpsd[fidx:]) * df
    return np.sqrt(SNRsq)


def whiten_data(data, duration, sample_rate, psd, flag='td'):
    if flag == 'td':
        win = tukey(duration * sample_rate, alpha=1.0 / 8.0)
        xf = np.fft.rfft(win * data)
    else:
        xf = data
    idx = np.argwhere(psd > 0.0)
    invpsd = np.zeros(psd.size)
    invpsd[idx] = 1.0 / psd[idx]
    xf *= np.sqrt(2.0 * invpsd / sample_rate)
    xf[0] = 0.0
    if flag == 'td':
        return np.fft.irfft(xf)
    return xf


# ================================================================
# get_fmin — 含 leading-order tidal 修正
# ================================================================
def get_fmin(M, eta, dt, verbose, Lambda_tilde=0.0):
    """
    根据 chirp time 反推信号进入观测窗口时的最低频率。

    在原始 point-particle PN 表达式基础上，加入 leading-order tidal
    修正；当 Lambda_tilde = 0 时，退化为 point-particle 情形。
    """
    M_SI = M * MSUN_SI

    def dtchirp(f):
        v = ((G_SI / C_SI ** 3) * M_SI * np.pi * f) ** (1.0 / 3.0)
        # point-particle PN terms (to 2PN)
        pp = (v ** (-8.0)
              + ((743.0 / 252.0) + 11.0 * eta / 3.0) * v ** (-6.0)
              - (32.0 * np.pi / 5.0) * v ** (-5.0)
              + ((3058673.0 / 508032.0) + 5429.0 * eta / 504.0
                 + (617.0 / 72.0) * eta ** 2) * v ** (-4.0))
        t_pp = (5.0 / (256.0 * eta)) * (G_SI / C_SI ** 3) * M_SI * pp

        # leading-order tidal correction to chirp time
        # Enters at 5PN relative order (Newtonian tidal, -1PN absolute)
        # delta_t_tidal ~ -(39/2) * Lambda_tilde * (G M / c^3) / (256 eta) * v^5
        if Lambda_tilde > 0.0:
            t_tidal = -(39.0 / 2.0) * Lambda_tilde * (
                (G_SI / C_SI ** 3) * M_SI / (256.0 * eta)
            ) * v ** 5
        else:
            t_tidal = 0.0

        return t_pp + t_tidal - dt

    fmin = brentq(dtchirp, 1.0, 2000.0, xtol=1e-6)
    if verbose:
        print('{}: signal enters segment at {:.2f} Hz '
              '(Lambda_tilde={:.1f})'.format(time.asctime(), fmin, Lambda_tilde))
    return fmin


def make_bbh(hp, hc, fs, ra, dec, psi, det, verbose):
    """探测器响应 — 与 BBH 版一致。"""
    tvec = np.arange(len(hp)) / float(fs)
    resp = AntennaResponse(det, ra, dec, psi, scalar=True, vector=True, times=0.0)
    Fp = resp.plus
    Fc = resp.cross
    ht = hp * Fp + hc * Fc

    frDetector = lalsimulation.DetectorPrefixToLALDetector(det)
    tdelay = lal.TimeDelayFromEarthCenter(frDetector.location, ra, dec, 0.0)
    if verbose:
        print('{}: computed {} time delay = {}'.format(
            time.asctime(), det, tdelay))

    ht_tck = interpolate.splrep(tvec, ht, s=0)
    hp_tck = interpolate.splrep(tvec, hp, s=0)
    hc_tck = interpolate.splrep(tvec, hc, s=0)
    tnew = tvec + tdelay
    new_ht = interpolate.splev(tnew, ht_tck, der=0, ext=1)
    new_hp = interpolate.splev(tnew, hp_tck, der=0, ext=1)
    new_hc = interpolate.splev(tnew, hc_tck, der=0, ext=1)

    return new_ht, new_hp, new_hc


# ================================================================
# Strict tidal waveform generation
# ================================================================
def _require_lalsim_attr(name):
    if not hasattr(lalsimulation, name):
        raise RuntimeError(
            'Strict BNS mode requires lalsimulation.{} '
            'but current build does not provide it.'.format(name))
    return getattr(lalsimulation, name)


APPROXIMANT_NAME = 'IMRPhenomD_NRTidalv2'


def _compute_lambda_tilde(m1, m2, lam1, lam2):
    """
    有效潮汐可形变度，用于 get_fmin 的 tidal 修正。

    定义: Eq. (5) of Favata, PRL 112, 101101 (2014).
    """
    M = m1 + m2
    return (16.0 / 13.0) * (
        (m1 + 12.0 * m2) * m1 ** 4 * lam1
        + (m2 + 12.0 * m1) * m2 ** 4 * lam2
    ) / M ** 5


def build_bns_lal_dict(m1, m2, eos_name=BNS_EOS_NAME):
    """
    为 BNS 波形生成构建包含潮汐参数的 LALDict。

    用 LAL TOV solver 精确计算两颗星的 Lambda1, Lambda2。
    两颗星使用同一个 EOS。
    """
    insert_lambda1 = _require_lalsim_attr(
        'SimInspiralWaveformParamsInsertTidalLambda1')
    insert_lambda2 = _require_lalsim_attr(
        'SimInspiralWaveformParamsInsertTidalLambda2')

    lambda1 = compute_lambda_from_eos(m1, eos_name)
    lambda2 = compute_lambda_from_eos(m2, eos_name)

    lal_dict = lal.CreateDict()
    insert_lambda1(lal_dict, lambda1)
    insert_lambda2(lal_dict, lambda2)
    return lal_dict, lambda1, lambda2


def choose_bns_approximant():
    return _require_lalsim_attr(APPROXIMANT_NAME)


def _generate_td_waveform_strict(par, fs, N, verbose):
    approximant = choose_bns_approximant()

    # 直接使用 par 中已计算好的 lambda1, lambda2，避免重复 TOV 查询
    insert_lambda1 = _require_lalsim_attr(
        'SimInspiralWaveformParamsInsertTidalLambda1')
    insert_lambda2 = _require_lalsim_attr(
        'SimInspiralWaveformParamsInsertTidalLambda2')
    lal_dict = lal.CreateDict()
    insert_lambda1(lal_dict, par.lambda1)
    insert_lambda2(lal_dict, par.lambda2)

    dist = 1e6 * lal.PC_SI

    f_low = float(np.clip(par.fmin, 5.0, 2048.0))
    min_f_low = 5.0

    if verbose:
        print('{}: strict BNS: EOS={}, approx={}'.format(
            time.asctime(), par.eos_name, APPROXIMANT_NAME))
        print('{}: tidal lambdas = ({:.1f}, {:.1f})'.format(
            time.asctime(), par.lambda1, par.lambda2))

    while f_low >= min_f_low:
        try:
            hp, hc = lalsimulation.SimInspiralChooseTDWaveform(
                par.m1 * lal.MSUN_SI, par.m2 * lal.MSUN_SI,
                0, 0, 0, 0, 0, 0,
                dist,
                par.iota, par.phi, 0,
                0, 0,
                1.0 / fs,
                f_low, f_low,
                lal_dict,
                approximant)
        except Exception as exc:
            raise RuntimeError(
                'Strict BNS waveform generation failed with {} at '
                'f_low={:.2f} Hz: {}'.format(APPROXIMANT_NAME, f_low, exc))

        if hp.data.length > 2 * N:
            if verbose:
                print('{}: waveform accepted at f_low={:.2f} Hz '
                      '(length={})'.format(
                          time.asctime(), f_low, hp.data.length))
            return hp.data.data, hc.data.data, f_low, par.lambda1, par.lambda2

        if verbose:
            print('{}: waveform too short at f_low={:.2f} Hz '
                  '(length={}), lowering'.format(
                      time.asctime(), f_low, hp.data.length))
        f_low -= 1.0

    raise RuntimeError(
        'Strict BNS waveform with {} did not reach required length > {} '
        'samples. Tried down to f_low={:.1f} Hz.'.format(
            APPROXIMANT_NAME, 2 * N, min_f_low))


# ================================================================
# BNS-specific public API
# ================================================================
def gen_masses_bns(m_min=BNS_MASS_MIN, m_max=BNS_MASS_MAX, verbose=True):
    """BNS 质量采样: 均匀分布于 [m_min, m_max] Msun。"""
    if verbose:
        print('{}: BNS uniform mass [{}, {}] Msun'.format(
            time.asctime(), m_min, m_max))
    m12 = np.random.uniform(m_min, m_max, 2)
    m12 = np.sort(m12)[::-1]  # m1 >= m2
    eta = m12[0] * m12[1] / (m12[0] + m12[1]) ** 2
    mc = np.sum(m12) * eta ** (3.0 / 5.0)
    return m12, mc, eta


def gen_par_bns(fs, T_obs, beta=[0.75, 0.95], verbose=True):
    """生成一组 BNS 参数，包含从 TOV 计算的潮汐参数。"""
    m12, mc, eta = gen_masses_bns(verbose=verbose)
    M = np.sum(m12)
    if verbose:
        print('{}: BNS masses = {:.4f}, {:.4f} (Mc = {:.4f})'.format(
            time.asctime(), m12[0], m12[1], mc))

    iota = np.arccos(-1.0 + 2.0 * np.random.rand())
    psi = 2.0 * np.pi * np.random.rand()
    phi = 2.0 * np.pi * np.random.rand()
    ra = 2.0 * np.pi * np.random.rand()
    dec = np.arcsin(-1.0 + 2.0 * np.random.rand())

    low_idx, high_idx = convert_beta(beta, fs, T_obs)
    if low_idx == high_idx:
        idx = low_idx
    else:
        idx = int(np.random.randint(low_idx, high_idx, 1)[0])

    # 计算 tidal 参数（用于 fmin 修正和存档）
    lam1 = compute_lambda_from_eos(m12[0], BNS_EOS_NAME)
    lam2 = compute_lambda_from_eos(m12[1], BNS_EOS_NAME)
    Lambda_tilde = _compute_lambda_tilde(m12[0], m12[1], lam1, lam2)

    sidx = int(0.5 * fs * T_obs * (safe - 1.0) / safe)
    fmin = get_fmin(M, eta, int(idx - sidx) / fs, verbose,
                    Lambda_tilde=Lambda_tilde)

    par = bnsparams(mc, M, eta, m12[0], m12[1], ra, dec,
                    iota, phi, psi, idx, fmin, None, None,
                    eos_name=BNS_EOS_NAME, lambda1=lam1, lambda2=lam2)
    return par


def gen_bns(fs, T_obs, psds, snr=1.0, dets=['H1'], beta=[0.75, 0.95],
            par=None, verbose=True):
    """
    生成 strict tidal BNS 时域信号。
    """
    N = T_obs * fs
    if par is None:
        raise ValueError('gen_bns requires a non-None par object.')

    orig_hp, orig_hc, used_f_low, lambda1, lambda2 = \
        _generate_td_waveform_strict(par=par, fs=fs, N=N, verbose=verbose)

    ref_idx = np.argmax(orig_hp ** 2 + orig_hc ** 2)

    win = np.zeros(N)
    tempwin = tukey(int((16.0 / 15.0) * N / safe), alpha=1.0 / 8.0)
    win[int((N - tempwin.size) / 2):
        int((N - tempwin.size) / 2) + tempwin.size] = tempwin

    ndet = len(psds)
    ts = np.zeros((ndet, N))
    hp_out = np.zeros((ndet, N))
    hc_out = np.zeros((ndet, N))
    intsnr = []

    for j, (det, psd) in enumerate(zip(dets, psds)):
        ht_shift, hp_shift, hc_shift = make_bbh(
            orig_hp, orig_hc, fs, par.ra, par.dec, par.psi, det, verbose)
        ht_temp = ht_shift[int(ref_idx - par.idx):]
        hp_temp = hp_shift[int(ref_idx - par.idx):]
        hc_temp = hc_shift[int(ref_idx - par.idx):]
        if len(ht_temp) < N:
            ts[j, :len(ht_temp)] = ht_temp
            hp_out[j, :len(ht_temp)] = hp_temp
            hc_out[j, :len(ht_temp)] = hc_temp
        else:
            ts[j, :] = ht_temp[:N]
            hp_out[j, :] = hp_temp[:N]
            hc_out[j, :] = hc_temp[:N]

        ts[j, :] *= win
        hp_out[j, :] *= win
        hc_out[j, :] *= win
        intsnr.append(get_snr(ts[j, :], T_obs, fs, psd.data.data, par.fmin))

    intsnr = np.array(intsnr)
    scale = snr / np.sqrt(np.sum(intsnr ** 2))
    ts *= scale
    hp_out *= scale
    hc_out *= scale
    intsnr *= scale

    if verbose:
        print('{}: strict BNS network SNR = {}'.format(time.asctime(), snr))
        print('{}: EOS={}, lambdas=({:.1f}, {:.1f})'.format(
            time.asctime(), BNS_EOS_NAME, lambda1, lambda2))

    return ts, hp_out, hc_out


# ================================================================
# 数据集模拟 — 接口与原版完全一致
# ================================================================
def sim_data_bns(fs, T_obs, snr=1.0, dets=['H1'], Nnoise=25, size=1000,
                 beta=[0.75, 0.95], verbose=True):
    yval = []
    ts = []
    par = []
    nclass = 2
    npclass = int(size / float(nclass))
    ndet = len(dets)
    psds = [gen_psd(fs, T_obs, op='AdvDesign', det=d) for d in dets]

    # 纯噪声
    for x in xrange(npclass):
        if verbose:
            print('{}: noise {}/{}'.format(time.asctime(), x + 1, npclass))
        ts_new = np.array([gen_noise(fs, T_obs, psd.data.data)
                           for psd in psds]).reshape(ndet, -1)
        ts.append(np.array([whiten_data(t, T_obs, fs, psd.data.data)
                            for t, psd in zip(ts_new, psds)
                            ]).reshape(ndet, -1))
        par.append(None)
        yval.append(0)

    # 信号 + 噪声
    cnt = npclass
    while cnt < size:
        par_new = gen_par_bns(fs, T_obs, beta=beta, verbose=verbose)
        ts_new, _, _ = gen_bns(fs, T_obs, psds, snr=snr, dets=dets,
                               beta=beta, par=par_new, verbose=verbose)

        for _ in xrange(Nnoise):
            ts_noise = np.array([gen_noise(fs, T_obs, psd.data.data)
                                 for psd in psds]).reshape(ndet, -1)
            ts.append(np.array([whiten_data(t, T_obs, fs, psd.data.data)
                                for t, psd in zip(ts_noise + ts_new, psds)
                                ]).reshape(ndet, -1))
            par.append(par_new)
            yval.append(1)
            cnt += 1
            if cnt >= size:
                break

    ts = np.array(ts)[:size]
    yval = np.array(yval)[:size]
    par = par[:size]

    idx = np.random.permutation(size)
    temp = [par[i] for i in idx]
    return [ts[idx], yval[idx]], temp
