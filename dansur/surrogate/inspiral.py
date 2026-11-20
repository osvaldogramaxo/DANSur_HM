"""IMRPhenomT(HM) inspiral in batched torch, for hybridising DANSur with a long inspiral.

Port of the inspiral region of IMRPhenomTHM (Estelles et al. 2022, arXiv:2012.11923) from
LALSimulation v6.2.0 (LALSimIMRPhenomTHM_internals.c, LALSimIMRPhenomTHM_fits.c), default
inspiral reconstruction. The PN coefficients, fits and phase ansatz were transcribed
mechanically from the C source. Dimensionless throughout: t in M, h = r h / M, float64.

Modes follow Blanchet et al. 2008 (arXiv:0802.1249), the convention of DANSur's mode phases:
    h_lm = 2 eta sqrt(16 pi / 5) x H_lm(x) exp(-i m phi22 / 2),   x = (omega22 / 2)^(2/3),
with |H_22| for the (2,2), as IMRPhenomTHM does. (2,2), (2,1), (3,3), (4,4) and (5,5) use the
IMRPhenomTHM PN amplitudes; (3,2) and (4,3), which IMRPhenomTHM lacks, use the 3PN nonspinning
amplitudes of Blanchet et al. 2008 eq. (9.4), plus the 0.5PN spin terms of SEOBNRv4HM
(LALSimIMRSpinEOBFactorizedWaveform.c, rho32 and f43).

ponytail: the calibrated amplitude terms of IMRPhenomTHM (x^4 ... x^5, collocation points at
-2000, -250, -150 M) are left out; they need the merger model and change the amplitude by
<~1e-3 before -3000 M. Add them if the inspiral is used closer to merger.
"""
import math

import numpy as np
import torch

EULER_GAMMA = 0.5772156649015329
_THETA_CP = np.array([0.33, 0.45, 0.55, 0.65, 0.75, 0.82])  # (2,2) frequency collocation points in theta
# The collocation matrix of IMRPhenomTSetPhase22Coefficients (rows theta_i^8 ... theta_i^13) only
# depends on the fixed thetas, so it is inverted once.
_AINV = np.linalg.inv(_THETA_CP[:, None] ** np.arange(8, 14))
MODES = ((2, 2), (2, 1), (3, 3), (4, 4), (5, 5), (3, 2), (4, 3))


def _pw(a, b):
    return a ** b


def _log(x):
    return torch.log(x) if torch.is_tensor(x) else np.log(x)


def _taylort3_coeffs(eta, chi1, chi2, delta):
    c = {}
    c['omega1PN'] = 0.27641369047619047 + (11*eta)/32.
    c['omega1halfPN'] = (-19*(chi1 + chi2)*eta)/80. + (-113*(chi2*(-1 + delta) - chi1*(1 + delta)) - 96*math.pi)/320.
    c['omega2PN'] = (1855099 + 1714608*chi2*chi2*(-1 + delta) - 1714608*chi1*chi1*(1 + delta))/1.4450688e7 + ((56975 + 61236*chi1*chi1 - 119448*chi1*chi2 + 61236*chi2*chi2)*eta)/258048. + (371*eta*eta)/2048.
    c['omega2halfPN'] = (-17*(chi1 + chi2)*eta*eta)/128. + (-146597*(chi2*(-1 + delta) - chi1*(1 + delta)) - 46374*math.pi)/129024. + (eta*(-2*(chi1*(1213 - 63*delta) + chi2*(1213 + 63*delta)) + 117*math.pi))/2304.
    c['omega3PN'] = -2.499258364444952 - (16928263*chi1*chi1)/1.376256e8 - (16928263*chi2*chi2)/1.376256e8 - (16928263*chi1*chi1*delta)/1.376256e8 + (16928263*chi2*chi2*delta)/1.376256e8 + ((-2318475 + 18767224*chi1*chi1 - 54663952*chi1*chi2 + 18767224*chi2*chi2)*eta*eta)/1.376256e8 + (235925*eta*eta*eta)/1.769472e6 + (107*EULER_GAMMA)/280. - (6127*chi1*math.pi)/12800. - (6127*chi2*math.pi)/12800. - (6127*chi1*delta*math.pi)/12800. + (6127*chi2*delta*math.pi)/12800. + (53*math.pi*math.pi)/200. + (eta*(632550449425 + 35200873512*chi1*chi1 - 28527282000*chi1*chi2 + 9605339856*chi1*chi1*delta - 1512*chi2*chi2*(-23281001 + 6352738*delta) + 34172264448*(chi1 + chi2)*math.pi - 22912243200*math.pi*math.pi))/1.040449536e11 + (107*math.log(2))/280.
    c['omega3halfPN'] = (-12029*(chi1 + chi2)*eta*eta*eta)/92160. + (eta*eta*(507654*chi1*chi2*chi2 - 838782*chi2*chi2*chi2 + chi2*(-840149 + 507654*chi1*chi1 - 870576*delta) + chi1*(-840149 - 838782*chi1*chi1 + 870576*delta) + 1701228*math.pi))/1.548288e7 + (eta*(218532006*chi1*chi2*chi2*(-1 + delta) - 1134*chi2*chi2*chi2*(-206917 + 71931*delta) - chi2*(1496368361 - 429508815*delta + 218532006*chi1*chi1*(1 + delta)) + chi1*(-1496368361 - 429508815*delta + 1134*chi1*chi1*(206917 + 71931*delta)) - 144*(488825 + 923076*chi1*chi1 - 1782648*chi1*chi2 + 923076*chi2*chi2)*math.pi))/1.8579456e8 + (-6579635551*chi2*(-1 + delta) + 535759434*chi2*chi2*chi2*(-1 + delta) - chi1*(-6579635551 + 535759434*chi1*chi1)*(1 + delta) + (-565550067 - 465230304*chi2*chi2*(-1 + delta) + 465230304*chi1*chi1*(1 + delta))*math.pi)/1.30056192e9
    return c


def _fits(eta, S, dchi, delta):
    """TaylorT3 t0 and the (2,2) frequency collocation points at theta = 0.45 ... 0.82."""
    tt0 = _pw(eta,-1)*((-20.74399646637014 - 106.27711276502542*eta)*_pw(1 + 0.6516016033332481*eta,-1) + 0.0012450290074562259*dchi*delta*(1 - 4.701633367918768e6*eta)*_pw(eta,2) - 111.5049997379579*dchi*delta*(1 + 19.95458485773613*eta)*S*_pw(eta,2) + 1204.6829118499857*(1 - 4.025474056585855*eta)*_pw(dchi,2)*_pw(eta,3) + S*(338.7318821277009 - 1553.5891860091408*eta + 19614.263378999745*_pw(eta,2) - 156449.78737303324*_pw(eta,3) + 577363.3090369126*_pw(eta,4) - 802867.433363341*_pw(eta,5)) + (-55.75053935847546 - 290.36341163610575*eta + 7873.7667183299345*_pw(eta,2) - 43585.59040070178*_pw(eta,3) + 87229.84668746481*_pw(eta,4) - 32469.263449695136*_pw(eta,5))*_pw(S,2) + (-102.8269343111326 + 5121.845705262981*eta - 93026.46878769135*_pw(eta,2) + 650989.6793529999*_pw(eta,3) - 1.8846061037110784e6*_pw(eta,4) + 1.861602620702142e6*_pw(eta,5))*_pw(S,3) + (-7.294950933078567 + 314.24955197427136*eta - 3751.8509582195657*_pw(eta,2) + 21205.339564205595*_pw(eta,3) - 46448.94771114493*_pw(eta,4) + 20310.512558558552*_pw(eta,5))*_pw(S,4) + (97.22312282683716 - 4556.60375328623*eta + 76308.73046927384*_pw(eta,2) - 468784.4188333802*_pw(eta,3) + 998692.0246600509*_pw(eta,4) - 322905.9042578296*_pw(eta,5))*_pw(S,5))
    cps = [
        -0.014968864336704284*dchi*delta*(1 - 1.942061808318584*eta)*_pw(eta,2) + 0.0017312772309375462*dchi*delta*(1 - 0.07106994121956058*eta)*S*_pw(eta,2) + S*(0.0019208448318368731 - 0.0013579968243452476*eta - 0.0033501404728414627*_pw(eta,2) + 0.008914420175326192*_pw(eta,3)) + 6.687615165457298e-6*_pw(dchi,2)*_pw(eta,3) + (0.02104073275966069 + 717.1534194224539*eta + 85.37320237350282*_pw(eta,2) + 12.789214868358362*_pw(eta,3) - 16.00243777208413*_pw(eta,4))*_pw(1 + 32934.586638893634*eta,-1) + (-8.306810248117731e-6 + 0.00009918593182087119*eta - 0.003805916669791129*_pw(eta,2) + 0.009854209286892323*_pw(eta,3))*_pw(S,2) + (-5.578836442449699e-6 - 0.0030378960591856616*eta + 0.03746366675135751*_pw(eta,2) - 0.10298471015315146*_pw(eta,3))*_pw(S,3) + (0.00004425141111368952 - 0.0008702073302258368*eta + 0.006538604805919268*_pw(eta,2) - 0.01578597166324495*_pw(eta,3))*_pw(S,4) + (-0.000019469656288570753 + 0.002969863931498354*eta - 0.03643271052162611*_pw(eta,2) + 0.09959495981802587*_pw(eta,3))*_pw(S,5) + (-0.000042037164406446896 + 0.0007336074135429041*eta - 0.005603356997202016*_pw(eta,2) + 0.013439843000090702*_pw(eta,3))*_pw(S,6),
        -0.04486391236129559*dchi*delta*(1 - 1.8997912248414794*eta)*_pw(eta,2) - 0.003531802135161727*dchi*delta*(1 - 8.001211450141325*eta)*S*_pw(eta,2) + S*(0.0061664395419698285 - 0.0040934633081508905*eta - 0.009180337242551828*_pw(eta,2) + 0.020338583755834694*_pw(eta,3)) + 0.00006524644306613066*_pw(dchi,2)*_pw(eta,3) + _pw(1 - 3.2125452791404148*eta,-1)*(0.03711511661217631 - 0.10663782888636487*eta - 0.09963406984414182*_pw(eta,2) + 0.6597367702009397*_pw(eta,3) - 2.777344875144891*_pw(eta,4) + 4.220674345359693*_pw(eta,5)) + (0.00044302547647888445 + 0.000424246501303979*eta - 0.01394093576260671*_pw(eta,2) + 0.02634851560709597*_pw(eta,3))*_pw(S,2) + (0.00011582043047950321 - 0.008282652950117982*eta + 0.08965067576998058*_pw(eta,2) - 0.23963885130463913*_pw(eta,3))*_pw(S,3) + (0.0006123158975881322 - 0.007809160444435783*eta + 0.028517174579539676*_pw(eta,2) - 0.03717957419042746*_pw(eta,3))*_pw(S,4) + (-0.0000885530893214531 + 0.005939789043536808*eta - 0.07106551435109858*_pw(eta,2) + 0.1891131957235774*_pw(eta,3))*_pw(S,5) + (-0.0005110853374341054 + 0.0038762476596420855*eta + 0.005094077179675256*_pw(eta,2) - 0.047971766995287136*_pw(eta,3))*_pw(S,6),
        -0.10196878573773932*dchi*delta*(1 - 1.8918584778973513*eta)*_pw(eta,2) - 0.018820536453940443*dchi*delta*(1 - 3.7307154599131183*eta)*S*_pw(eta,2) - 0.00013162098437956188*_pw(dchi,2)*_pw(eta,3) + S*(0.0145572994468378 - 0.0017482433991394227*eta - 0.10299007619034371*_pw(eta,2) + 0.4581039376357615*_pw(eta,3) - 0.7123678787549022*_pw(eta,4)) + (0.05489007025458171 + 5.852073438961151*eta + 2.74597705533403*_pw(eta,2) + 4.834336623113389*_pw(eta,3) - 26.931994454691022*_pw(eta,4) + 57.67035368809743*_pw(eta,5))*_pw(1 + 105.52132834236778*eta,-1) + (0.003001211395915229 + 0.0017929418998452987*eta - 0.13776590125456148*_pw(eta,2) + 0.7471133710854526*_pw(eta,3) - 1.3620323111858437*_pw(eta,4))*_pw(S,2) + (0.001143282743686261 - 0.05793457776296727*eta + 0.7841331051705482*_pw(eta,2) - 3.4936244160305323*_pw(eta,3) + 4.802357041496856*_pw(eta,4))*_pw(S,3) + (0.0009168588840889624 - 0.03261437094899735*eta + 0.3472881896838799*_pw(eta,2) - 1.3634383958859384*_pw(eta,3) + 1.7313939586675267*_pw(eta,4))*_pw(S,4) + (-0.0002794014744432316 + 0.055911057147527664*eta - 0.8686311380514122*_pw(eta,2) + 4.096191294930781*_pw(eta,3) - 6.009676060669872*_pw(eta,4))*_pw(S,5) + (-0.0005046018052528331 + 0.029804593053788925*eta - 0.3792653361049425*_pw(eta,2) + 1.6366976231421981*_pw(eta,3) - 2.26904099961476*_pw(eta,4))*_pw(S,6),
        -0.1831889759662071*dchi*delta*(1 - 1.8484261527766557*eta)*_pw(eta,2) - 0.07586202965525136*dchi*delta*(1 - 3.2918162656371983*eta)*S*_pw(eta,2) + 0.0019259052728265817*_pw(dchi,2)*_pw(eta,3) + S*(0.02685637375751212 + 0.013341664908359861*eta - 0.3057217933283597*_pw(eta,2) + 1.395763446325911*_pw(eta,3) - 2.2559396974665376*_pw(eta,4)) + (0.0725639467287476 + 12.39400068457852*eta + 12.907450928972402*_pw(eta,2) - 7.422660061864399*_pw(eta,3) + 66.32985901506036*_pw(eta,4) - 117.85875779454518*_pw(eta,5))*_pw(1 + 168.63492460136445*eta,-1) + (0.0087781653701194 + 0.006944161553839352*eta - 0.3301149078235105*_pw(eta,2) + 1.6835714783903248*_pw(eta,3) - 2.950404929598742*_pw(eta,4))*_pw(S,2) + (0.0037229746496019625 - 0.17155338099487646*eta + 2.5881802140836774*_pw(eta,2) - 13.14710199375518*_pw(eta,3) + 21.366803256010915*_pw(eta,4))*_pw(S,3) + (0.00278507305662002 - 0.12475855143364532*eta + 1.8640209516178643*_pw(eta,2) - 10.117078727717564*_pw(eta,3) + 17.94244821676711*_pw(eta,4))*_pw(S,4) + (0.0010273954584773936 + 0.1713357629442166*eta - 3.017249223460983*_pw(eta,2) + 15.855096360798678*_pw(eta,3) - 26.444621592311933*_pw(eta,4))*_pw(S,5) + (-0.00012207946532225968 + 0.11709700788855186*eta - 2.0950821618097026*_pw(eta,2) + 11.925324501640054*_pw(eta,3) - 21.683978511818076*_pw(eta,4))*_pw(S,6),
        -0.2508206617297265*dchi*delta*(1 - 1.861010982421798*eta)*_pw(eta,2) - 0.1392163711259171*dchi*delta*(1 - 3.2669366465555796*eta)*S*_pw(eta,2) + 0.0023126403170013045*_pw(dchi,2)*_pw(eta,3) + S*(0.036750064163293766 + 0.036904343404333906*eta - 0.5238739410356437*_pw(eta,2) + 2.3292117112945223*_pw(eta,3) - 3.654184701923543*_pw(eta,4)) + (0.08373610487663233 + 6.301736487754372*eta + 9.03911386193751*_pw(eta,2) + 4.91153188278086*_pw(eta,3))*_pw(1 + 72.64820846804257*eta,-1) + (0.014963449678540705 + 0.008354571522567225*eta - 0.41723078020683*_pw(eta,2) + 2.2007932082378785*_pw(eta,3) - 4.245354787320365*_pw(eta,4))*_pw(S,2) + (0.005706180633326235 - 0.15748500622007494*eta + 2.3477109912232845*_pw(eta,2) - 11.413877195221694*_pw(eta,3) + 17.033120593116756*_pw(eta,4))*_pw(S,3) + (0.003890296981717687 - 0.15985471334551038*eta + 2.560312006077997*_pw(eta,2) - 14.400920672743332*_pw(eta,3) + 26.10406142567958*_pw(eta,4))*_pw(S,4) + (0.005305988847210204 + 0.10869207132210629*eta - 2.4201307115268875*_pw(eta,2) + 12.544899744864924*_pw(eta,3) - 19.550600837316903*_pw(eta,4))*_pw(S,5) + (0.002917248769788225 + 0.11851143848720952*eta - 2.6640023622893416*_pw(eta,2) + 15.993378498844761*_pw(eta,3) - 29.752144941054446*_pw(eta,4))*_pw(S,6),
    ]
    return tt0, cps


def _taylort3(theta, c):
    th2 = theta * theta
    poly = (1 + c['omega1PN'] * th2 + c['omega1halfPN'] * th2 * theta + c['omega2PN'] * th2 * th2
            + c['omega2halfPN'] * th2 * th2 * theta
            + (c['omega3PN'] + 107 * _log(theta) / 280) * th2 ** 3 + c['omega3halfPN'] * th2 ** 3 * theta)
    return theta ** 3 / 4 * poly


def _omega22_lal(c, eta, t):
    """IMRPhenomTInspiralOmegaAnsatz22 as in LAL: reference for the self-check of PhenomTInspiral."""
    theta = (-eta * t / 5) ** (-1 / 8)
    ext = c['C1'] + theta * (c['C2'] + theta * (c['C3'] + theta * (c['C4'] + theta * (c['C5'] + theta * c['C6']))))
    return _taylort3(theta, c) + theta ** 11 / 4 * ext


def _phase22_lal(c, eta, t):
    """IMRPhenomTInspiralPhaseAnsatz22 as transcribed from LAL (without the constant phOffInsp):
    reference for the self-check of the polynomial form used by PhenomTInspiral."""
    thetabar = (-eta * t) ** (-1 / 8)
    return -(_pw(5,-0.625)*_pw(eta,-2)*_pw(t,-1)*_pw(thetabar,-7)*(3*(-107 + 280*c['omega3PN'])*_pw(5,0.75) + 321*torch.log(thetabar*_pw(5,0.125))*_pw(5,0.75) + 420*c['omega3halfPN']*thetabar*_pw(5,0.875) + 56*(25*c['C1'] + 3*eta*t)*_pw(thetabar,2) + 1050*c['C2']*_pw(5,0.125)*_pw(thetabar,3) + 280*(3*c['C3'] + eta*c['omega1PN']*t)*_pw(5,0.25)*_pw(thetabar,4) + 140*(5*c['C4'] + 3*eta*c['omega1halfPN']*t)*_pw(5,0.375)*_pw(thetabar,5) + 120*(5*c['C5'] + 7*eta*c['omega2PN']*t)*_pw(5,0.5)*_pw(thetabar,6) + 525*c['C6']*_pw(5,0.625)*_pw(thetabar,7) + 105*eta*c['omega2halfPN']*t*torch.log(-t)*_pw(5,0.625)*_pw(thetabar,7)))/84.


def _horner(P, x):
    """sum_k P[:, k] x^k for coefficients P (n, K) and x (n, T), by fused multiply-adds."""
    r = P[:, -1:]
    for k in range(P.shape[1] - 2, -1, -1):
        r = torch.addcmul(P[:, k:k + 1], r, x)
    return r


def _pn_amp_coeffs(l, m, eta, delta, chi1, chi2):
    """H_lm = sum_k coeffs[k] x^(k/2) + amplog x^3 log(16 x), in the Blanchet et al. 2008 convention."""
    if (l, m) == (2, 2):
        coeffs = [
            (1.0) + 1j * (0.0),
            (0.0) + 1j * (0.0),
            (-2.5476190476190474 + (55*eta)/42.) + 1j * (0.0),
            ((-2*chi1)/3. - (2*chi2)/3. - (2*chi1*delta)/(3.*((1 - delta)/2. + (1 + delta)/2.)) + (2*chi2*delta)/(3.*((1 - delta)/2. + (1 + delta)/2.)) + (2*chi1*eta)/3. + (2*chi2*eta)/3. + 2*math.pi) + 1j * (0.0),
            (-1.437169312169312 + _pw(chi1,2)/2. + _pw(chi2,2)/2. + (_pw(chi1,2)*delta)/2. - (_pw(chi2,2)*delta)/2. - (1069*eta)/216. - _pw(chi1,2)*eta + 2*chi1*chi2*eta - _pw(chi2,2)*eta + (2047*_pw(eta,2))/1512.) + 1j * (0.0),
            (- (107*math.pi)/21. + (34*eta*math.pi)/21.) + 1j * (-24.*eta),
            (41.78634662956092 - (278185*eta)/33264. - (20261*_pw(eta,2))/2772. + (114635*_pw(eta,3))/99792. - (856*EULER_GAMMA)/105. + (2*_pw(math.pi,2))/3. + (41*eta*_pw(math.pi,2))/96.) + 1j * ((428./105)*math.pi),
            ((-2173*math.pi)/756. - (2495*eta*math.pi)/378. + (40*_pw(eta,2)*math.pi)/27.) + 1j * ((14333*eta)/162. - (4066*_pw(eta,2))/945.),
        ]
        amplog = -428/105.
    elif (l, m) == (2, 1):
        coeffs = [
            (0.0) + 1j * (0.0),
            (delta/3.) + 1j * (0.0),
            (-chi1/4. + chi2/4. - (chi1*delta)/(4.*((1 - delta)/2. + (1 + delta)/2.)) - (chi2*delta)/(4.*((1 - delta)/2. + (1 + delta)/2.))) + 1j * (0.0),
            (-17*delta/84. + (5*delta*eta)/21.) + 1j * (0.0),
            ((79*chi1)/84. - (79*chi2)/84. + (79*chi1*delta)/84. + (79*chi2*delta)/84. - (43*chi1)/(42.*((1 - delta)/2. + (1 + delta)/2.)) + (43*chi2)/(42.*((1 - delta)/2. + (1 + delta)/2.)) - (43*chi1*delta)/(42.*((1 - delta)/2. + (1 + delta)/2.)) - (43*chi2*delta)/(42.*((1 - delta)/2. + (1 + delta)/2.)) - (139*chi1*eta)/84. + (139*chi2*eta)/84. - (139*chi1*delta*eta)/84. - (139*chi2*delta*eta)/84. + (86*chi1*eta)/(21.*((1 - delta)/2. + (1 + delta)/2.)) - (86*chi2*eta)/(21.*((1 - delta)/2. + (1 + delta)/2.)) + (43*chi1*delta*eta)/(21.*((1 - delta)/2. + (1 + delta)/2.)) + (43*chi2*delta*eta)/(21.*((1 - delta)/2. + (1 + delta)/2.)) + (delta*math.pi)/3.) + 1j * (-delta/6. - (2*delta*math.log(2))/3.),
            ((-43*delta)/378. - (509*delta*eta)/378. + (79*delta*_pw(eta,2))/504.) + 1j * (0.0),
            ((-17*delta*math.pi)/84. + (delta*eta*math.pi)/14.) + 1j * (-((-17*delta)/168. + (353*delta*eta)/84. - (17*delta*math.log(2))/42. + (delta*eta*math.log(2))/7.)),
            (0.0) + 1j * (0.0),
        ]
        amplog = 0.0
    elif (l, m) == (3, 3):
        coeffs = [
            (0.0) + 1j * (0.0),
            (0.7763237542601484*delta) + 1j * (0.0),
            (0.0) + 1j * (0.0),
            (-3.1052950170405937*delta + 1.5526475085202969*delta*eta) + 1j * (0.0),
            (-(-0.5822428156951114*chi1 + 0.5822428156951114*chi2 - 7.316679009572791*delta - 0.5822428156951114*chi1*delta - 0.5822428156951114*chi2*delta + (1.3585665699552598*chi1)/((1 - delta)/2. + (1 + delta)/2.) - (1.3585665699552598*chi2)/((1 - delta)/2. + (1 + delta)/2.) + (1.3585665699552598*chi1*delta)/((1 - delta)/2. + (1 + delta)/2.) + (1.3585665699552598*chi2*delta)/((1 - delta)/2. + (1 + delta)/2.) + 1.7467284470853341*chi1*eta - 1.7467284470853341*chi2*eta + 1.7467284470853341*chi1*delta*eta + 1.7467284470853341*chi2*delta*eta - (5.434266279821039*chi1*eta)/((1 - delta)/2. + (1 + delta)/2.) + (5.434266279821039*chi2*eta)/((1 - delta)/2. + (1 + delta)/2.) - (2.7171331399105196*chi1*delta*eta)/((1 - delta)/2. + (1 + delta)/2.) - (2.7171331399105196*chi2*delta*eta)/((1 - delta)/2. + (1 + delta)/2.))) + 1j * (-1.371926598204461*delta),
            (-(-0.08680711070363478*delta + 8.647776123213047*delta*eta - 2.0866641516022777*delta*_pw(eta,2))) + 1j * (0.0),
            (0.0) + 1j * (0.0),
            (0.0) + 1j * (0.0),
        ]
        amplog = 0.0
    elif (l, m) == (4, 4):
        coeffs = [
            (0.0) + 1j * (0.0),
            (0.0) + 1j * (0.0),
            (0.751248226425348*(1 - 3*eta)) + 1j * (0.0),
            (0.0) + 1j * (0.0),
            (-4.049910893365739 + 14.489984730901032*eta - 5.9758381647470875*_pw(eta,2)) + 1j * (0.0),
            (0.751248226425348*(4*math.pi - 12*eta*math.pi)) + 1j * (0.751248226425348*(-2.854822555520438 + 13.189467666561313*eta)),
            (-(-8*_pw(0.7142857142857143,0.5)*(5.338016983016983 - (1088119*eta)/28600. + (146879*_pw(eta,2))/2340. - (226097*_pw(eta,3))/17160.))/9.) + 1j * (0.0),
            (0.0) + 1j * (0.0),
        ]
        amplog = 0.0
    elif (l, m) == (5, 5):
        coeffs = [
            (0.0) + 1j * (0.0),
            (0.0) + 1j * (0.0),
            (0.0) + 1j * (0.0),
            (0.8013768943966973*delta*(1 - 2*eta)) + 1j * (0.0),
            (0.0) + 1j * (0.0),
            (0.8013768943966973*delta*(-6.743589743589744 + (688*eta)/39. - (256*_pw(eta,2))/39.)) + 1j * (0.0),
            (12.58799882096634*delta - 25.175997641932675*delta*eta) + 1j * (-3.0177162096765713*delta + 12.454250695829877*delta*eta),
            (0.0) + 1j * (0.0),
        ]
        amplog = 0.0
    elif (l, m) == (3, 2):
        # Blanchet et al. 2008 eq. (9.4); 0.5PN spin term from rho32 of SEOBNRv4HM
        # (LALSimIMRSpinEOBFactorizedWaveform.c), large near q ~ 1.5 where 1 - 3 eta is small
        s = math.sqrt(5 / 7) / 3
        coeffs = [0, 0, s * (1 - 3 * eta), s * 4 * eta * (chi1 + chi2) / 2,
                  s * (-193 / 90 + 145 / 18 * eta - 73 / 18 * eta ** 2),
                  s * (2 * math.pi * (1 - 3 * eta) + 1j * (-3 + 66 / 5 * eta)),
                  s * (-1451 / 3960 - 17387 / 3960 * eta + 5557 / 220 * eta ** 2 - 5341 / 1320 * eta ** 3), 0]
        return coeffs, 0
    elif (l, m) == (4, 3):
        # Blanchet et al. 2008 eq. (9.4) (LALSimInspiralPNMode43 has the opposite sign of the 1PN
        # term, which overshoots SXS by 10-20%); 0.5PN spin term from f43 of SEOBNRv4HM
        s = -1j * 9 / (4 * math.sqrt(70))
        chiS, chiA = (chi1 + chi2) / 2, (chi1 - chi2) / 2
        l32 = math.log(3 / 2)
        coeffs = [0, 0, 0, s * delta * (1 - 2 * eta), s * -2.5 * eta * (chiA - delta * chiS),
                  s * delta * (-39 / 11 + 1267 / 132 * eta - 131 / 33 * eta ** 2),
                  s * delta * (3 * math.pi * (1 - 2 * eta) + 1j * (-32 / 5 + eta * (16301 / 810 - 12 * l32) + 6 * l32)), 0]
        return coeffs, 0
    else:
        raise ValueError(f'mode {(l, m)} not available')
    # IMRPhenomTHM stores its amplitudes rotated by exp(-i phoff) (eq. 13 of the THM paper); undo it
    rot = {(2, 2): 1, (2, 1): 1j, (3, 3): -1j, (4, 4): -1, (5, 5): 1j}[(l, m)]
    return [ck * rot for ck in coeffs], amplog * rot


def _params(q, chi1, chi2):
    """NumPy float64 arrays (n,) of q, chi1, chi2 (arrays, tensors or scalars), with eta and delta."""
    q, chi1, chi2 = (np.atleast_1d(v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)).astype(np.float64)
                     for v in (q, chi1, chi2))
    m1, m2 = q / (1 + q), 1 / (1 + q)
    return q, chi1, chi2, m1 * m2, m1 - m2


class PNAmplitudes:
    """PN mode amplitudes H_lm = sum_k c_k x^(k/2) + c_8 x^3 log(16 x) of _pn_amp_coeffs for a batch:
    the real and imaginary parts of the nonzero coefficients, set up in NumPy and copied to `device`
    once, as (n, 1) tensors. Shared by PhenomTInspiral and DANSur's inter-mode phase offsets."""

    def __init__(self, q, chi1, chi2, device='cpu', modes=MODES):
        q, chi1, chi2, eta, delta = _params(q, chi1, chi2)
        n, entries = len(q), []
        for lm in modes:
            coeffs, amplog = _pn_amp_coeffs(*lm, eta, delta, chi1, chi2)
            for k, ck in enumerate(list(coeffs) + [amplog]):  # k = 8: the x^3 log(16 x) term
                if not (isinstance(ck, (int, float, complex)) and ck == 0):
                    entries.append((lm, k, ck))
        Z = np.empty((n, 2 * len(entries)))
        for j, (_, _, ck) in enumerate(entries):
            Z[:, 2 * j], Z[:, 2 * j + 1] = np.real(ck), np.imag(ck)
        keep = np.flatnonzero(np.any(Z != 0, axis=0))  # parts that vanish for the whole batch are skipped
        K = torch.as_tensor(Z[:, keep]).to(device)
        col = {int(i): K[:, j:j + 1] for j, i in enumerate(keep)}
        self.terms = {lm: [] for lm in modes}
        for j, (lm, k, _) in enumerate(entries):
            re, im = col.get(2 * j), col.get(2 * j + 1)
            if re is not None or im is not None:
                self.terms[lm].append((k, re, im))

    def needs_log(self, modes):
        return any(k == 8 for lm in modes for k, _, _ in self.terms[lm])

    def evaluate(self, pw, modes):
        """{(l, m): [Re H_lm, Im H_lm]} for powers pw[k] = x^(k/2), k = 1 ... 7, and pw[8] = x^3 log(16 x);
        None for a part that vanishes, an (n, 1) column for a constant one."""
        out = {}
        for lm in modes:
            parts = [None, None]
            for k, *cs in self.terms[lm]:
                for j, ck in enumerate(cs):
                    if ck is not None:
                        acc = parts[j]
                        if k == 0:
                            parts[j] = ck if acc is None else acc + ck
                        else:
                            parts[j] = ck * pw[k] if acc is None else torch.addcmul(acc, ck, pw[k])
            out[lm] = parts
        return out


class PhenomTInspiral:
    """IMRPhenomT(HM) inspiral for a batch of aligned-spin binaries.

    q >= 1, chi1, chi2: arrays or tensors of shape (n,), chi1 the spin of the heavier black hole.
    The per-binary coefficients are set up in NumPy (a few hundred operations on length-n arrays,
    cheaper than as many small torch kernels on any device) and moved to `device` (default: that
    of q) in one copy; everything evaluated per sample runs in torch there.
    Valid well before merger: the inspiral region ends at t = -5 / (eta 0.81^8), about -110 M at
    q = 1 and -270 M at q = 8.
    """

    def __init__(self, q, chi1, chi2, device=None):
        if device is None:
            device = q.device if torch.is_tensor(q) else 'cpu'
        q, chi1, chi2, eta, delta = _params(q, chi1, chi2)
        n = len(q)
        m1, m2 = q / (1 + q), 1 / (1 + q)
        S = (m1 ** 2 * chi1 + m2 ** 2 * chi2) / (m1 ** 2 + m2 ** 2)
        c = _taylort3_coeffs(eta, chi1, chi2, delta)
        tt0, cps = _fits(eta, S, chi1 - chi2, delta)
        # Collocation system of IMRPhenomTSetPhase22Coefficients: the first point is TaylorT3 with the
        # fitted t0 at theta = 0.33.
        theta_ini = (eta * (tt0 + 5 / (eta * 0.33 ** 8)) / 5) ** (-1 / 8)
        omega_cp = np.stack([_taylort3(theta_ini, c)] + cps, -1)
        b = 4 / _THETA_CP ** 3 * (omega_cp - _taylort3(_THETA_CP, {k: v[:, None] for k, v in c.items()}))
        C = b @ _AINV.T
        c.update({f'C{k + 1}': C[:, k] for k in range(6)})
        self._c = c  # NumPy, for the self-check

        cols = []  # every per-binary coefficient, packed into one (n, K) array

        def put(v):
            cols.append(v)
            return len(cols) - 1

        i_eta, i_fac0 = put(eta), put(2 * eta * math.sqrt(16 * math.pi / 5))
        # omega22 = theta^3/4 [TaylorT3 polynomial + theta^8 (C1 + ... + C6 theta^5)] with
        # theta = 5^(1/8) thetabar: coefficients of thetabar^3 ... thetabar^16, and of thetabar^9 log(thetabar)
        s5 = 5 ** (1 / 8)
        w = {3: 1.0, 5: c['omega1PN'], 6: c['omega1halfPN'], 7: c['omega2PN'], 8: c['omega2halfPN'],
             9: c['omega3PN'] + 107 / 280 * math.log(5) / 8, 10: c['omega3halfPN'],
             **{10 + k: c[f'C{k}'] for k in range(1, 7)}}
        i_w = [put(s5 ** k / 4 * w.get(k, 0.0)) for k in range(3, 17)]
        i_wL = put(s5 ** 9 / 4 * 107 / 280)
        # phase22 = (1/eta) [sum_k ph[k + 5] thetabar^k (k = -5 ... 8) + (phL[0] + phL[1] thetabar) log(thetabar)],
        # derived exactly from the LAL phase ansatz by the generator (t = -thetabar^-8 / eta)
        logeta = np.log(eta)
        ph, phL = [0.0] * 14, [0.0] * 2
        ph[0] = -2/5*5**(3/8)
        ph[2] = -2/3*5**(5/8)*c['omega1PN']
        ph[3] = -5**(3/4)*c['omega1halfPN']
        ph[4] = -2*5**(7/8)*c['omega2PN']
        ph[5] = (5/4)*logeta*c['omega2halfPN']
        ph[6] = 10*5**(1/8)*c['omega3PN'] - 107/28*5**(1/8) + (107/224)*5**(1/8)*math.log(5)
        ph[7] = 5*5**(1/4)*c['omega3halfPN']
        ph[8] = (10/3)*5**(3/8)*c['C1']
        ph[9] = (5/2)*math.sqrt(5)*c['C2']
        ph[10] = 2*5**(5/8)*c['C3']
        ph[11] = (5/3)*5**(3/4)*c['C4']
        ph[12] = (10/7)*5**(7/8)*c['C5']
        ph[13] = (25/4)*c['C6']
        phL[0] = 10*c['omega2halfPN']
        phL[1] = (107/28)*5**(1/8)
        i_ph = [put(v / eta) for v in ph]
        i_phL = [put(v / eta) for v in phL]
        Z = np.empty((n, len(cols)))
        for i, v in enumerate(cols):
            Z[:, i] = v
        K = torch.as_tensor(Z).to(device)
        col = lambda i: K[:, i:i + 1]
        self.eta, self.fac0 = col(i_eta), col(i_fac0)
        self.W, self.WL = K[:, i_w[0]:i_w[-1] + 1], col(i_wL)
        self.P, self.PL0, self.PL1 = K[:, i_ph[0]:i_ph[-1] + 1], col(i_phL[0]), col(i_phL[1])
        self.amp = PNAmplitudes(q, chi1, chi2, device)

    def _thetabar(self, t):
        tb = (-self.eta * t) ** (-1 / 8)
        return tb, torch.log(tb)

    def _omega(self, tb, L):
        tb3 = tb * tb * tb
        return torch.addcmul(tb3 * _horner(self.W, tb), self.WL * L, tb3 * tb3 * tb3)

    def _phase(self, tb, L):
        tb2 = tb * tb
        return _horner(self.P, tb) / (tb2 * tb2 * tb) + torch.addcmul(self.PL0, self.PL1, tb) * L

    def omega22(self, t):
        """(2,2) angular frequency at times t (n, T) in M."""
        return self._omega(*self._thetabar(t))

    def phase22(self, t):
        """(2,2) phase, h22 ~ exp(-i phase22), up to a constant: the analytic integral of omega22."""
        return self._phase(*self._thetabar(t))

    def omega_phase22(self, t):
        """(omega22, phase22) at times t, sharing thetabar."""
        tb, L = self._thetabar(t)
        return self._omega(tb, L), self._phase(tb, L)

    def modes(self, t, modes=MODES, phase_shift=0.0, return_x=False):
        """r h_lm / M at times t (n, T) in M, with phase22 -> phase22 + phase_shift; also x if return_x."""
        tb, L = self._thetabar(t)
        x = (self._omega(tb, L) / 2) ** (2 / 3)
        sx = x.sqrt()
        pw = [None, sx, x]  # x^(k/2), built once and shared by the modes
        for k in range(3, 8):
            pw.append(pw[k - 2] * x)
        if self.amp.needs_log(modes):
            pw.append(pw[6] * torch.log(16 * x))
        e1 = torch.polar(torch.ones_like(x), -0.5 * (self._phase(tb, L) + phase_shift))  # exp(-i m psi) = e1^m
        em = {1: e1}
        A = self.fac0 * x
        out = {}
        for (l, m), parts in self.amp.evaluate(pw, modes).items():
            Hr, Hi = (torch.zeros_like(x) if p is None else p.expand_as(x) for p in parts)
            for j in range(2, m + 1):
                if j not in em:
                    em[j] = em[j - 1] * e1
            if (l, m) == (2, 2):
                out[(l, m)] = (A * torch.hypot(Hr, Hi)) * em[2]
            else:
                out[(l, m)] = torch.complex(A * Hr, A * Hi) * em[m]
        return (out, x) if return_x else out


if __name__ == '__main__':
    # self-check: the polynomial forms match the LAL ansatzes, the phase integrates the frequency,
    # and the frequency rises monotonically
    insp = PhenomTInspiral(torch.tensor([1., 3., 8.]), torch.tensor([0., .5, -.8]), torch.tensor([0., -.3, .8]))
    t = torch.linspace(-2e5, -1000, 400001, dtype=torch.float64).expand(3, -1)
    w, ph = insp.omega_phase22(t)
    cT = {k: torch.as_tensor(v)[:, None] for k, v in insp._c.items()}
    w_lal, ph_lal = _omega22_lal(cT, insp.eta, t), _phase22_lal(cT, insp.eta, t)
    err_w = ((w - w_lal) / w_lal).abs().max().item()
    d = ph - ph_lal  # equal up to a constant; relative to each row's phase range (the phase crosses zero)
    err_ph = ((d - d[:, :1]).abs().amax(-1) / ph_lal.abs().amax(-1)).max().item()
    assert err_w < 1e-13 and err_ph < 1e-12, (err_w, err_ph)
    dph = (ph[:, 2:] - ph[:, :-2]) / (t[:, 2:] - t[:, :-2])
    err = ((dph - w[:, 1:-1]) / w[:, 1:-1]).abs().max().item()
    assert err < 1e-7, err
    assert (w.diff(dim=-1) > 0).all()
    h = insp.modes(t[:, -10:])
    assert all(torch.isfinite(v).all() for v in h.values()) and len(h) == 7
    print(f'ok: omega22 vs LAL {err_w:.1e}, phase22 vs LAL {err_ph:.1e} (relative), '
          f'max |dphase/dt - omega| / omega = {err:.1e}')
