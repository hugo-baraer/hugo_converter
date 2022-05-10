"""
  EoR_research/project_driver.py

  Author : Hugo Baraer
  Supervision by : Prof. Adrian Liu
  Affiliation : Cosmic dawn group at McGill University
  Date of creation : 2021-09-21

  This module is the driver and interacts between 21cmFast and the modules computing the require fields and parameters.
"""


#import classic python librairies
import py21cmfast as p21c
from py21cmfast import plotting
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import emcee
from scipy import signal
import imageio
import corner
import powerbox as pbox
#import this project's modules
import z_re_field as zre
import Gaussian_testing as gauss
import FFT
import statistical_analysis as sa
import plot_params as pp
import statistics
from numpy import array
#import pymks

b_0_range = np.linspace(7.8,8, 20)
data_dict = {'Z_re': [], 'Heff': [], "medians": [], "a16":[], "a50":[], "a84":[], "b16":[], "b50":[], "b84":[], "k16":[], "k50":[], "k84":[], "p16":[], "p50":[], "p84":[]}
#a = {'Z_re': [7.659, 7.67369125, 7.6883824999999995, 7.70307375, 7.717765, 7.73245625, 7.7471475, 7.76183875, 7.77653, 7.7912212499999995, 7.8059125, 7.82060375, 7.835295, 7.84998625, 7.8646775, 7.87936875, 7.89406, 7.90875125, 7.9234425], 'Heff': [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0], 'medians': [array([0.65633357, 1.13302524, 0.19000859, 0.01183023]), array([0.64576227, 1.15543329, 0.17715654, 0.01437947]), array([0.67691815, 1.09312683, 0.21583314, 0.01455461]), array([0.6752209 , 1.0641355 , 0.22413127, 0.01366672]), array([0.66536696, 1.23568873, 0.17915538, 0.01459614]), array([0.66717185, 1.19448805, 0.18883506, 0.01401136]), array([0.67313161, 1.1071573 , 0.21302732, 0.01477765]), array([0.684663  , 1.1097716 , 0.22271708, 0.01278479]), array([0.67683172, 1.12216674, 0.21298786, 0.01405916]), array([0.6689276 , 1.18530804, 0.19491084, 0.0128609 ]), array([0.66112154, 1.19259987, 0.18733177, 0.01615764]), array([0.65586766, 1.27802502, 0.16964287, 0.01756909]), array([0.68676841, 1.11226874, 0.22642664, 0.01302317]), array([0.68102479, 1.11866278, 0.21770371, 0.01628345]), array([0.66897034, 1.15137335, 0.20648509, 0.01465579]), array([0.68272586, 1.16930905, 0.21111293, 0.01721193]), array([0.65556955, 1.25045887, 0.17338502, 0.01620082]), array([0.67488905, 1.13114804, 0.21141054, 0.01410652]), array([0.67934353, 1.15266142, 0.2126821 , 0.01567662])], 'a16': [array([0.66275508]), array([0.66283754]), array([0.66273989]), array([0.66318314]), array([0.66204644]), array([0.66106077]), array([0.66087134]), array([0.66034991]), array([0.66128853]), array([0.66071001]), array([0.66298973]), array([0.66299789]), array([0.66229491]), array([0.66226768]), array([0.66296359]), array([0.66276658]), array([0.662696]), array([0.66459266]), array([0.66350941])], 'a50': [array([0.67328676]), array([0.67356235]), array([0.67343102]), array([0.67380041]), array([0.67292841]), array([0.67206419]), array([0.67189976]), array([0.67168163]), array([0.67251374]), array([0.67208598]), array([0.67411638]), array([0.67415753]), array([0.67328307]), array([0.6736553]), array([0.67397266]), array([0.67392598]), array([0.67429098]), array([0.67538524]), array([0.67447562])], 'a84': [array([0.68402869]), array([0.68452891]), array([0.68448559]), array([0.68513786]), array([0.68421461]), array([0.68322093]), array([0.68321283]), array([0.68357778]), array([0.68417513]), array([0.68379166]), array([0.68547094]), array([0.68594514]), array([0.68488719]), array([0.68483611]), array([0.6852938]), array([0.68588891]), array([0.68586375]), array([0.68682786]), array([0.68603387])], 'b16': [array([1.06910568]), array([1.07057775]), array([1.07468592]), array([1.07714974]), array([1.08009283]), array([1.08581984]), array([1.08821385]), array([1.09104305]), array([1.09461445]), array([1.0965575]), array([1.09691619]), array([1.09959611]), array([1.1052787]), array([1.10820594]), array([1.11085723]), array([1.11382003]), array([1.1162071]), array([1.11868851]), array([1.1233135])], 'b50': [array([1.10642429]), array([1.1088792]), array([1.11269513]), array([1.11547939]), array([1.11997266]), array([1.12674448]), array([1.12873794]), array([1.13282439]), array([1.13589522]), array([1.14095697]), array([1.14034518]), array([1.1423048]), array([1.14761969]), array([1.15070626]), array([1.1537779]), array([1.15729872]), array([1.15922527]), array([1.16119395]), array([1.16699338])], 'b84': [array([1.14739263]), array([1.15037293]), array([1.15391406]), array([1.15837159]), array([1.1648174]), array([1.17039874]), array([1.17428325]), array([1.18023203]), array([1.18159975]), array([1.18611539]), array([1.18582301]), array([1.1884392]), array([1.19307738]), array([1.19792698]), array([1.19912313]), array([1.2038108]), array([1.20664406]), array([1.20744524]), array([1.21162177])], 'k16': [array([0.19314339]), array([0.19320977]), array([0.19276635]), array([0.19274463]), array([0.19132301]), array([0.18959545]), array([0.18974094]), array([0.18834716]), array([0.18888881]), array([0.18804985]), array([0.19044447]), array([0.19014475]), array([0.1891374]), array([0.18900598]), array([0.18929009]), array([0.18883787]), array([0.18873978]), array([0.19025929]), array([0.18903042])], 'k50': [array([0.20888282]), array([0.20930066]), array([0.20859824]), array([0.20862246]), array([0.20736051]), array([0.20610697]), array([0.20587513]), array([0.20524322]), array([0.20568859]), array([0.20480924]), array([0.20704044]), array([0.20685525]), array([0.20553272]), array([0.20557432]), array([0.20568606]), array([0.20542304]), array([0.20546694]), array([0.20633293]), array([0.20497927])], 'k84': [array([0.22533483]), array([0.22620173]), array([0.22496304]), array([0.22596953]), array([0.22473749]), array([0.22324409]), array([0.2230081]), array([0.22325169]), array([0.22334984]), array([0.22301601]), array([0.22460043]), array([0.22487692]), array([0.22318328]), array([0.22288184]), array([0.22297027]), array([0.22342032]), array([0.22324725]), array([0.22377224]), array([0.22289331])], 'p16': [array([0.01234804]), array([0.01241626]), array([0.0125079]), array([0.01249352]), array([0.01287843]), array([0.01296399]), array([0.01312566]), array([0.01349316]), array([0.01342523]), array([0.01362958]), array([0.01316354]), array([0.01333008]), array([0.01340165]), array([0.01329708]), array([0.01323791]), array([0.01345447]), array([0.01344233]), array([0.01331245]), array([0.01338056])], 'p50': [array([0.01350936]), array([0.01354932]), array([0.01367498]), array([0.01365508]), array([0.01406086]), array([0.01415357]), array([0.01434423]), array([0.01469906]), array([0.01467465]), array([0.01493367]), array([0.01439197]), array([0.014582]), array([0.01467625]), array([0.01453673]), array([0.01444738]), array([0.01469072]), array([0.01467476]), array([0.01456518]), array([0.01464436])], 'p84': [array([0.0148331]), array([0.01492307]), array([0.0150267]), array([0.01501271]), array([0.01546258]), array([0.01559009]), array([0.01576735]), array([0.01614248]), array([0.01613459]), array([0.01642447]), array([0.01583076]), array([0.01599424]), array([0.01610552]), array([0.01597702]), array([0.01582158]), array([0.01609107]), array([0.01614743]), array([0.01599435]), array([0.01608297])]}
# a = {'Z_re': [6.0, 6.105263157894737, 6.2105263157894735, 6.315789473684211, 6.421052631578947, 6.526315789473684, 6.631578947368421, 6.7368421052631575, 6.842105263157895, 6.947368421052632, 7.052631578947368, 7.157894736842105, 7.2631578947368425, 7.368421052631579, 7.473684210526316, 7.578947368421053, 7.684210526315789, 7.789473684210526, 7.894736842105263], 'Heff': [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0], 'medians': [array([0.63180813, 0.7538829 , 0.20515303, 0.0101804 ]), array([0.63761888, 0.81791775, 0.19975391, 0.01132231]), array([0.62993268, 0.86236725, 0.18260956, 0.00989073]), array([0.6459764 , 0.85037064, 0.20099963, 0.01468858]), array([0.63860633, 0.91363411, 0.18357672, 0.01009687]), array([0.64456359, 0.90963741, 0.19704261, 0.01320558]), array([0.65043168, 0.8928169 , 0.20866583, 0.01055495]), array([0.64147009, 0.90023065, 0.19931342, 0.01291619]), array([0.66058177, 0.95501285, 0.20675364, 0.01251466]), array([0.64544875, 1.03969763, 0.17658289, 0.01299319]), array([0.65552901, 1.00814268, 0.19726053, 0.01270558]), array([0.67777985, 1.02967296, 0.2151892 , 0.014464  ]), array([0.66512924, 0.98861275, 0.21426133, 0.01308504]), array([0.66877652, 0.97425996, 0.2252668 , 0.01397767]), array([0.66449909, 1.09424445, 0.19748583, 0.01491009]), array([0.67409341, 1.05520403, 0.21804893, 0.01179999]), array([0.66598743, 1.11350009, 0.20197486, 0.01335685]), array([0.6789967 , 1.14328629, 0.20941141, 0.01590717]), array([0.66318364, 1.27215986, 0.17697339, 0.01587511])], 'a16': [array([0.61948184]), array([0.61902322]), array([0.62573052]), array([0.63157448]), array([0.63341018]), array([0.63458052]), array([0.63716065]), array([0.63526045]), array([0.6381871]), array([0.63781794]), array([0.6402342]), array([0.65010113]), array([0.65726945]), array([0.65873829]), array([0.66013826]), array([0.66172454]), array([0.66365009]), array([0.6610348]), array([0.66328407])], 'a50': [array([0.62919085]), array([0.62893834]), array([0.63612783]), array([0.64161601]), array([0.64356613]), array([0.64467122]), array([0.64741914]), array([0.64654106]), array([0.64882798]), array([0.64912697]), array([0.65140969]), array([0.66031208]), array([0.66755025]), array([0.66896523]), array([0.67065295]), array([0.67254777]), array([0.67397661]), array([0.67246166]), array([0.67423927])], 'a84': [array([0.63913438]), array([0.63989041]), array([0.64639012]), array([0.65173881]), array([0.6536327]), array([0.65520166]), array([0.65822276]), array([0.65750697]), array([0.66000878]), array([0.6607566]), array([0.66274169]), array([0.67110786]), array([0.67820199]), array([0.6794233]), array([0.68153985]), array([0.68319983]), array([0.6849129]), array([0.68449193]), array([0.68554856])], 'b16': [array([0.76730385]), array([0.78734303]), array([0.80315938]), array([0.82132693]), array([0.84107041]), array([0.86062553]), array([0.87885268]), array([0.90191415]), array([0.92111338]), array([0.94308959]), array([0.96350279]), array([0.97765332]), array([0.99047417]), array([1.01263198]), array([1.03168363]), array([1.05320851]), array([1.0735421]), array([1.09428435]), array([1.11675467])], 'b50': [array([0.79532637]), array([0.81804316]), array([0.83247397]), array([0.84974718]), array([0.87034397]), array([0.89194628]), array([0.91130881]), array([0.93765846]), array([0.95607021]), array([0.98134033]), array([1.00070152]), array([1.0130189]), array([1.0251299]), array([1.04788072]), array([1.06796274]), array([1.09028369]), array([1.1114784]), array([1.13842306]), array([1.1602501])], 'b84': [array([0.82608842]), array([0.85048802]), array([0.86369124]), array([0.88168965]), array([0.9023693]), array([0.92564983]), array([0.94669377]), array([0.97593146]), array([0.99511101]), array([1.02383602]), array([1.04381812]), array([1.05136048]), array([1.06225838]), array([1.08616544]), array([1.10875508]), array([1.13033032]), array([1.15323475]), array([1.18505217]), array([1.20629516])], 'k16': [array([0.17447557]), array([0.17215596]), array([0.17827545]), array([0.18254331]), array([0.18309984]), array([0.18237443]), array([0.18341886]), array([0.17944467]), array([0.18100934]), array([0.17819953]), array([0.1794429]), array([0.18813093]), array([0.19449881]), array([0.19371384]), array([0.19366066]), array([0.1936255]), array([0.19342006]), array([0.18852413]), array([0.18920012])], 'k50': [array([0.18945793]), array([0.18771688]), array([0.19407495]), array([0.19810867]), array([0.19856818]), array([0.19809307]), array([0.19931445]), array([0.19606485]), array([0.19718989]), array([0.19541062]), array([0.19634767]), array([0.20363647]), array([0.21013466]), array([0.20963478]), array([0.20973488]), array([0.20913508]), array([0.20912811]), array([0.20548464]), array([0.20554926])], 'k84': [array([0.20558213]), array([0.20492655]), array([0.21090416]), array([0.21450562]), array([0.21467548]), array([0.21465286]), array([0.21644086]), array([0.21385008]), array([0.21467166]), array([0.21353426]), array([0.21402182]), array([0.22102632]), array([0.2265944]), array([0.22589762]), array([0.22670174]), array([0.22605414]), array([0.22600335]), array([0.22377332]), array([0.22266545])], 'p16': [array([0.00991272]), array([0.01048521]), array([0.01016312]), array([0.01023245]), array([0.01037325]), array([0.0107708]), array([0.01089966]), array([0.01201166]), array([0.01191853]), array([0.01261626]), array([0.01264764]), array([0.01178352]), array([0.01166292]), array([0.01162317]), array([0.01200255]), array([0.01214952]), array([0.01246164]), array([0.01355803]), array([0.01345433])], 'p50': [array([0.01084751]), array([0.01146098]), array([0.01110701]), array([0.01116649]), array([0.0113431]), array([0.01179262]), array([0.0118779]), array([0.0131272]), array([0.01301988]), array([0.01377067]), array([0.01381301]), array([0.01288115]), array([0.01274777]), array([0.01272418]), array([0.01309395]), array([0.01329437]), array([0.01363127]), array([0.01482019]), array([0.01470179])], 'p84': [array([0.01188899]), array([0.01255455]), array([0.01220805]), array([0.01230036]), array([0.01246378]), array([0.01296915]), array([0.01307548]), array([0.01442561]), array([0.01435092]), array([0.01513824]), array([0.01518782]), array([0.01414109]), array([0.01402415]), array([0.01399648]), array([0.01434049]), array([0.01459845]), array([0.01495823]), array([0.01629319]), array([0.01615835])]}
# fig, ax = plt.subplots()
# plt.errorbar(a['Z_re'], np.concatenate(a['b50']), yerr=(np.concatenate(a['b50'])-np.concatenate(a['b16']),np.concatenate(a['b84'])-np.concatenate(a['b50'])), ls = '')
# plt.scatter(a['Z_re'], np.concatenate(a['b50']), color = 'r')
# #plt.xlabel('Ionization efficiency')
# plt.xlabel(r'$\tilde{z_{re}}$')
# plt.ylabel(r'$b_0$ best fitted value')
# plt.show()
for zre_mean in b_0_range:
    #adjustable parameters to look out before running the driver
    #change the dimension of the box and see effect
    Heff = 30.0
    use_cache = True # uncomment this line to re-use field and work only on MCMC part
    box_dim = 143 #the desired spatial resolution of the box (corrected for Mpc/h instead of MPC to get the deried 100Mpc/h box size
    radius_thick = 2. #the radii thickness (will affect the number of bins and therefore points)
    box_len = 143 #int(143) #default value of 300
    user_params = {"HII_DIM": box_dim, "BOX_LEN": box_len, "DIM":box_len}
    cosmo_params = p21c.CosmoParams(SIGMA_8=0.8, hlittle=0.7, OMm= 0.27, OMb= 0.045)
    astro_params = p21c.AstroParams({ "HII_EFF_FACTOR": Heff }) #"HII_EFF_FACTOR":Heff, "M_TURN" : Heff
    flag_options = p21c.FlagOptions({"USE_MASS_DEPENDENT_ZETA": True})
    #add astro_params

    initial_conditions = p21c.initial_conditions(
    user_params = user_params,
    cosmo_params = cosmo_params,
    )

    #a = p21c.ionize_box(redshift=8.0, init_boxes = initial_conditions, astro_params = astro_params, write=False).z_re_box
    #xHI = p21c.ionize_box(redshift=7.0, zprime_step_factor=2.0, z_heat_max=15.0)

    #zre.generate_quick_zre_field(16, 1, 1, initial_conditions)

    #intialize a coeval cube at red shift z = z\bar
    #coeval = p21c.run_coeval(redshift=8.0,user_params={'HII_DIM': box_dim, 'BOX_LEN': box_len, "USE_INTERPOLATION_TABLES": False})

    if os.path.exists('b_mz.npy') and os.path.exists('bmz_errors.npy') and os.path.exists('kvalues.npy') and use_cache and os.path.exists('zre.npy'):
        #bmz_errors = np.load('bmz_errors.npy')
        #b_mz = np.load('b_mz.npy')
        kvalues = np.load('kvalues.npy')
        z_re = np.load('zre.npy')
        density_field = np.load('density.npy')
        overzre, zre_mean2 = zre.over_zre_field(z_re)
        #pp.ps_ion_map(overzre,2.0,143, plot = True)
    else :
        #plot the reionization redshift (test pursposes)
        # plotting.coeval_sliceplot(coeval, kind = 'z_re_box', cmap = 'jet')
        # plt.tight_layout()
        # plt.title('reionization redshift ')
        # plt.show()

        """
        it appears coeval has a Z_re component, which shows if yes or not, the pixel was ionized at that reshift. This means that the pixel value is either
        the redshift parameter entred in coeval, or -1 if it wasn't ionized at that redshift. 
        With these information, I could plot z_re as function of time, by looking at a bunch of redshifts.
        """




        """Test the FFT function for a 3D Gaussian field"""

        #start by generating a 3d gaussian field and plotting a slice of it (uncomment second line for plotting)
        gaussian_field, mu, std = gauss.generate_gaussian_field(box_dim)
        #gauss.plot_field(gaussian_field,int(box_dim//2), mu, std)

        #Gaussian_FFT for the 3D field, shift the field and plots with frquencies
        delta = 0.1 #an arbitrary desired time interval for the Gaussian
        X, Y, fft_gaussian_shifted = gauss.gaussian_fft(gaussian_field,delta,box_dim)
        #gauss.plot_ftt_field(fft_gaussian_shifted,int(box_dim//2), mu, std, X,Y)




        """Compute the over-redshfit or reionization and overdensity"""
        #Compute the reionization redshift from the module z_re
        #zre_range = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        #zre_range = np.linspace(7.99,7.5,1)
        #zre_range = [3,4,5,5.5,6,6.2,6.4,6.6,6.8,7.0,7.05, 7.1,7.15,7.2,7.25, 7.3,7.35,7.4,7.45,7.5,7.55,7.6,7.65,7.7,7.75,7.8,7.85, 7.9,7.95,8.0,8.05,8.1,8.15,8.2,8.25,8.3,8.35,8.4,8.45,8.5,8.55,8.6,8.65,8.7,8.75,8.8,8.85,8.9,8.95,9.0,9.05,9.1,9.15,9.2,9.25,9.3,9.35,9.4,9.45,9.5,9.55,9.6,9.65,9.7,9.75,9.8,9.85,9.9,9.95,10.0,10.1,10.2,10.3,10.4,10.5,10.6,10.7,10.8,10.9,11.0,11.1,11.2,11.3,11.4,11.5,11.6,11.7,11.8,11.9,12.0,12.2,12.4,12.6,12.8,13,13.25,13.5,13.75,14,14.25,14.5,14.75,15,15.5,16.5,17,17.5,18,18.5,19,19.5,20,20.5]
        zre_range = np.linspace(3.8,16,1000)
        z_re_box = zre.generate_zre_field(zre_range, initial_conditions,box_dim, astro_params, flag_options,comP_ionization_rate=False)
        overzre, zre_mean = zre.over_zre_field(z_re_box)
        np.save('./zre',z_re_box)
        #print(np.mean(overzre))
        #print(ionization_rate)
        #print(zre_mean)
        position_vec = np.linspace(-49,50,143)
        X, Y = np.meshgrid(position_vec, position_vec)
        fig, ax = plt.subplots()
        plt.contourf(X,Y,z_re_box[int(143//2)])
        plt.colorbar()
        ax.set_xlabel(r'[Mpc/h]')
        ax.set_ylabel(r'[Mpc/h]')
        plt.title(r'slice of a the over-redshift of reionization with 21cmFAST')
        plt.show()


        #Take and plot the Fourrier transform of the over-redshift along with it's frequnecy
        delta = 1 / (box_len / box_dim)
        Xz, Yz, overzre_fft, freqzre= FFT.compute_fft(overzre, delta, box_dim)

        #plot this F(over_zre(x))
        #FFT.plot_ftt_field(overzre_fft, int(box_dim//2), Xz, Yz, title = r'F($\delta_z$ (x)) at a pixel dimension of {}³'.format(box_dim))

        perturbed_field = p21c.perturb_field(redshift=zre_mean, init_boxes = initial_conditions)
        #coeval = p21c.run_coeval(redshift=zre_mean,user_params={'HII_DIM': box_dim, 'BOX_LEN': box_len, "USE_INTERPOLATION_TABLES": False})

        position_vec = np.linspace(-49,50,143)
        X, Y = np.meshgrid(position_vec, position_vec)
        # fig, ax = plt.subplots()
        # plt.contourf(X,Y,perturbed_field.density[int(box_dim//2)])
        # plt.colorbar()
        # ax.set_xlabel(r'[Mpc h⁻¹]')
        # ax.set_ylabel(r'[Mpc h⁻¹]')
        # plt.title(r'slice of a the over-redshift of reionization at the center with a pixel resolution of {} Mpc h⁻¹'.format('1'))
        # plt.show()

        np.save('./density', perturbed_field.density)

        Xd, Yd, overd_fft, freqd = FFT.compute_fft(perturbed_field.density, delta, box_dim)
        #FFT.plot_ftt_field(overd_fft, int(box_dim//2), Xd, Yd, title = r'F($\delta_m$ (x)) at a redshift of {} and a pixel dimension of {}³'.format(coeval.redshift,box_dim))
        freqs = np.fft.fftshift(np.fft.fftfreq(box_dim, d=delta))

        #polar_div = np.ones((box_dim,box_dim,box_dim))
        #polar_div = sa.cart2sphA(division)
        overd_fft = np.square(abs(overd_fft))
        overzre_fft = np.square(abs(overzre_fft))


        #plot the power of the field
        #FFT.plot_ftt_field(overzre_fft, int(box_dim//2), Xz, Yz, title = r'$|F(\delta_z (x))|^2$ at a pixel dimension of {}³'.format(box_dim))
        #wanted radius for plotting

        cx = int(box_dim // 2)
        #mask = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 == r**2

        #uncomment these lines to get a different amount of point repartition (more in the first pasrt and less in the second part
        # radii1 = np.linspace(0, np.sqrt((freqd/2) ** 2), num=int(3*np.sqrt((cx) ** 2) / int(radius_thick)))
        # radii2 = np.linspace(np.sqrt((freqd/2) ** 2), np.sqrt(3 * (freqd) ** 2), num=int(0.5*np.sqrt((cx) ** 2) / int(radius_thick)))
        # kvalues = np.concatenate((radii1[1:-1],radii2))


        kvalues = np.linspace(0,np.sqrt(3*(freqd)**2), num = int(np.sqrt(3*(cx)**2)/int(radius_thick)))
        kvalues = kvalues[1:]

        #xompute the average for the field
        values_zre, values_d, count_zre, count_d = sa.average_overk(box_dim,overzre_fft,overd_fft,radius_thick)

        overzre_fft_k = np.divide(values_zre,count_zre)
        overd_fft_k = np.divide(values_d,count_d)
        #xcompute the standard deviation of that average
        #sigmad, sigmazre = sa.average_std(box_dim,overzre_fft, overd_fft, radius_thick, overzre_fft_k, overd_fft_k, count_zre, count_d)


        #print(sigmad, sigmazre)
        #print(a[6,2], overzre_fft[6,2,int(box_dim//2)], overd_fft[6,2,int(box_dim//2)],int(box_dim//2))
        #xx=np.arange(0,len(oneD_div))
        #division = np.divide(overzre_fft[:,int(box_dim//2),int(box_dim//2)], overd_fft[:,int(box_dim//2),int(box_dim//2)])


        # fig, ax = plt.subplots()
        # plt.errorbar(kvalues, overd_fft_k, yerr = sigmad, linestyle = 'None',capsize=4, marker ='o')
        # #plt.legend()
        # ax.set_xlabel(r'$k [Mpc^{-1} h]$')
        # ax.set_ylabel(r'$\delta_m$ (k)')
        # plt.title(r'$\delta_m$ (k)) as a function of k ')
        # plt.show()


        # fig, ax = plt.subplots()
        # plt.errorbar(kvalues, overzre_fft_k, yerr = sigmazre, linestyle = 'None',capsize=4, marker ='o')
        # #plt.legend()
        # ax.set_xlabel(r'$k [Mpc^{-1}h]$')
        # ax.set_ylabel(r'$\delta_zre$ (k)')
        # plt.title(r'$\delta_zre$ (k) as a function of k ')
        # plt.show()


        #prim_basis = pymks.PrimitiveBasis(n_states=2)
        #X_ = prim_basis.discretize(overzre_fft_k)

        b_mz = np.sqrt(np.divide(overzre_fft_k,overd_fft_k))
        #bmz_errors = sa.compute_bmz_error(b_mz,overzre_fft_k,overd_fft_k,sigmad,sigmazre)
        #bmz = sa.compute_bmz(overzre_fft_k[1:],overd_fft_k[1:])
        #radii = radii[1:]
        #b_mz = b_mz[1:]
        #bmz_errors=bmz_errors[1:]

        np.save('./b_mz', b_mz)
        #np.save('./bmz_errors', bmz_errors)
        np.save('./kvalues', kvalues)


    """This section uses the redshfit of reionization field to evaluate the power spectrum with the premade pacakge"""

    overzre, zre_mean2 = zre.over_zre_field(z_re)
    #over_p, p_mean = zre.over_p_field(density_field)
    print(zre_mean)
    print(statistics.median(z_re.flatten()))
    z_reion_zre = np.load('zreion_for_Hugo.npz')['zreion']

    # redshifts = np.linspace(5, 12, 50)
    # position_vec = np.linspace(-49, 50, 143)
    # Xd, Yd = np.meshgrid(position_vec, position_vec)
    # filenames = []
    # movie_name = 'ion_diff.gif'
    # for i in redshifts:
    #     fig, ax = plt.subplots()
    #     plt.contourf(Xd, Yd, pp.ionization_map_diff(i, 143, z_re, z_reion_zre, plot = False)[int(143//2)],  cmap =  'RdBu', vmin=-1.0, vmax=1.0)
    #     #plt.title(r'slice of the ionization field at a redshift of {} '.format(i))
    #     plt.colorbar()
    #     ax.set_xlabel(r'$[\frac{Mpc}{\hbar}]$')
    #     ax.set_ylabel(r'$[\frac{Mpc}{\hbar}]$')
    #     plt.savefig('./ionization_map/ionization_field{}.png'.format(i))
    #     filenames.append('./ionization_map/ionization_field{}.png'.format(i))
    #     plt.close()
    # images = []
    # for filename in filenames:
    #     images.append(imageio.imread(filename))
    # imageio.mimsave(movie_name, images)
    # ionization_map_diff = pp.ionization_map_diff(8.011, 143, z_re, z_reion_zre)
    #reion_hist_zreion = pp.reionization_history(redshifts,z_reion_zre)
    #cmFast_hist = pp.reionization_history(redshifts,z_re)

    # fig, ax = plt.subplots()
    # # plt.scatter(redshifts,ionization_rate)
    # plt.plot(redshifts, reion_hist_zreion, label = 'z-reion')
    # plt.plot(redshifts, cmFast_hist, label = '21cmFAST')
    # ax.set_xlabel(r'z ')
    # ax.set_ylabel(r'$x_i(z)$')
    # plt.legend()
    # plt.title(r'ionization fraction as function of redshift')
    # plt.show()
    #pp.ionization_movie(np.linspace(5,15,20), z_re, 143, 'test_movie.gif')
    perturbed_field = p21c.perturb_field(redshift=zre_mean, init_boxes=initial_conditions)
    density_field = perturbed_field.density
    p_k_density, kbins_density = pbox.get_power(density_field, 100)
    p_k_zre, kbins_zre = pbox.get_power(overzre, 100)
    b_mz = np.sqrt(np.divide(p_k_zre, p_k_density))


    zre_pp = pp.ps_ion_map(overzre, 25, 143, logbins=True)
    den_pp = pp.ps_ion_map(density_field, 25, 143, logbins=True)

    delta = 0.1
    Xd, Yd, overd_fft, freqd = FFT.compute_fft(density_field, delta, box_dim)
    Xd, Yd, overzre_fft, freqd = FFT.compute_fft(overzre, delta, box_dim)
    cross_matrix = overd_fft * overzre_fft.conj().T
    b_mz1 = np.sqrt(np.divide(zre_pp, den_pp))
    b_mz2 = np.sqrt(np.divide(p_k_zre, p_k_density))

    values_cross, count_cross = sa.average_overk(143,cross_matrix,25, logbins=True)
    cross_pp = np.divide(values_cross, count_cross)
    # k_values = np.linspace(kbins_zre.min(), kbins_zre.max(), len(cross_pp))
    k_values = np.logspace(kbins_zre.min(), kbins_zre.max(), 25)
    # fig, ax = plt.subplots()
    # # plt.scatter(k_values, zre_pp, label = 'Pz with average')
    # # plt.scatter(k_values, den_pp, label='Pm with average')
    # # plt.scatter(k_values, p_k_zre, label='Pz with power_box')
    # # plt.scatter(k_values, p_k_density, label='Pm with power_box')
    # plt.scatter(k_values, b_mz2, label = 'bmz with power_box')
    # plt.scatter(k_values, b_mz1, label='bmz with average')
    # print(zre_pp,den_pp,p_k_zre, p_k_density, b_mz2-b_mz1)
    # #plt.scatter(k_values, cross_pp,label ='cross_pp')
    # # plt.errorbar(kvalues, b_mz, label = 'data fitting for',yerr = bmz_errors, linestyle = 'None',capsize=4, marker ='o') #xerr = sigmad[2:], yerr = sigmazre[2:], yerr = np.sqrt(bmz_errors)
    # # plt.title(r'$b_{zm}$ as a function of k ')
    # ax.set_ylabel(r'$Power spectrums $ ')
    # ax.set_xlabel(r'k [$Mpc^{-1} h$]')
    # plt.legend()
    # plt.xscale('log')
    # plt.show()
    
    #zre_pp = pp.ps_ion_map(overzre, radius_thick, 143)
    #den_pp = pp.ps_ion_map(density_field, radius_thick, 143)
    cross_cor = np.divide(cross_pp/8550986.582903482,np.sqrt(((zre_pp/8550986.582903482)*(den_pp/8550986.582903482))))
    #k_values = np.linspace(kbins_zre.min(), kbins_zre.max(), len(cross_cor))
    #perturbed_field = p21c.perturb_field(redshift=zre_mean, init_boxes=initial_conditions)
    #kvalues=kvalues[1:]
    #take out nan values in radii
    #nan_array = np.isnan(b_mz)
    #not_nan_array = ~ nan_array
    #bmz = b_mz[not_nan_array]
    #radii = radii[not_nan_array]
    """these lines plots the linear bias as a function of the kvalues"""
    fig, ax = plt.subplots()
    plt.scatter(k_values, cross_cor)
    #plt.errorbar(kvalues, b_mz, label = 'data fitting for',yerr = bmz_errors, linestyle = 'None',capsize=4, marker ='o') #xerr = sigmad[2:], yerr = sigmazre[2:], yerr = np.sqrt(bmz_errors)
    #plt.title(r'$b_{zm}$ as a function of k ')
    ax.set_ylabel(r'$r_{mz}$ ')
    ax.set_xlabel(r'k [$Mpc^{-1} h$]')
    plt.xscale('log')
    plt.show()
    #
    # ax.errorbar(kbins_zre, b_mz, yerr=(0.1 / (0.2*(kbins_zre/0.3)+1)**(0.6)), linestyle='None',
    #             capsize=4, marker='o')
    # plt.show()

    #plot the log version of this graph
    # fig, ax = plt.subplots()
    # #plt.plot(overd_fft_k[1:], y_plot_fit)
    # plt.errorbar(kvalues, b_mz, label = 'data fitting for',yerr = bmz_errors, linestyle = 'None',capsize=4, marker ='o') #xerr = sigmad[2:], yerr = sigmazre[2:], yerr = np.sqrt(bmz_errors)
    # plt.title(r'$b_{zm}$ as a function of k ')
    # ax.set_ylabel(r'$b_{mz}$')
    # plt.loglog()
    # # plt.xscale('log')
    # # plt.yscale('log')
    # #ax.set_ylim(bottom=0,top=1)
    # ax.set_xlabel(r'k')
    # plt.show()


    '''
    MCMC analysis and posterior distribution on the b_mz 
    '''

    #errs = np.ones_like(b_mz)*0.05

    #no b_mz fitting

    """
    #initialize the MCMC
    num_iter = 5000
    #ndim = 2 #excluding error
    ndim = 3 # number of parameters to fit for (including the error weighting parameter)
    nwalkers = 32
    
    #intial position for the 2 parameter fit
    #initial_pos = np.array((0.55, 0.03) + 0.015 * np.random.randn(nwalkers, ndim))
    #initial_pos = np.ones((ndim,nwalkers))
    #multiply by 0.5
    #initial_pos[0] = np.array((0.5) + 0.1 * np.random.randn(nwalkers))
    
    #add by
    #initial_pos[1] = np.array((0.1) + 0.02 * np.random.randn(nwalkers))
    #initial_pos = np.array((0.5, 1.7, 0.15) + 0.1 * np.random.randn(nwalkers, ndim))
    #bmz_errors = np.ones_like(b_mz)*0.03
    
    initial_pos = np.array((15, 1.3, 1) + 0.45 * np.random.randn(nwalkers, ndim))
    """

    num_iter = 5000
    #ndim = 2 #excluding error
    ndim = 4
    #ndim = 4 # number of parameters to fit for (including the error weighting parameter and b_0)
    nwalkers = 32

    #initial_pos = np.array((0.6, 0.8, 0.15, 0.5) + 0.02 * np.random.randn(nwalkers, ndim))
    #initial parameters for 4 dim
    initial_pos = np.ones((32,4))
    initial_pos[:,0] *= 0.7 +0.06 * np.random.randn(nwalkers)
    initial_pos[:,1] *= 0.8 + 0.1 * np.random.randn(nwalkers)
    initial_pos[:,2] *= 0.2 + 0.01 * np.random.randn(nwalkers)
    initial_pos[:,3] *= 0.01 + 0.005 * np.random.randn(nwalkers)

    # initial_pos = np.ones((32,3))
    # initial_pos[:,0] *= 0.9 +0.08 * np.random.randn(nwalkers)
    # initial_pos[:,1] *= 0.05 + 0.01 * np.random.randn(nwalkers)
    # initial_pos[:,2] *= 0.02 + 0.005 * np.random.randn(nwalkers)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, sa.log_post_bmz_b_errs, args=(kbins_zre, b_mz))
    sampler.run_mcmc(initial_pos, num_iter, progress=True);


    """The following lines need to be uncommented for plotting"""


    f, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = [r"$\alpha$", r"$b_0$", r"$k_0$", r"$p$"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        #ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("Step number");

    # f, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
    # samples = sampler.get_chain()
    # labels = [r"$\alpha$",  r"$k_0$", r"$p$"]
    # for i in range(ndim):
    #     ax = axes[i]
    #     ax.plot(samples[:, :, i], alpha=0.3)
    #     ax.set_xlim(0, len(samples))
    #     ax.set_ylabel(labels[i])
    #     # ax.yaxis.set_label_coords(-0.1, 0.5)
    #
    # axes[-1].set_xlabel("Step number");

    """
    f, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = [r"$\alpha$", r"$k_0$"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        #ax.yaxis.set_label_coords(-0.1, 0.5)
    
    axes[-1].set_xlabel("Step number");
    """

    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    fig = corner.corner(flat_samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
    inds = np.random.randint(len(flat_samples), size=100)

    best_walker = np.argmax(np.max(sampler.lnprobability,axis=1))
    best_params = samples[-1][np.argmax(sampler.lnprobability.T[best_walker])]


    # zre_mean = 8.015
    data_dict['medians'].append(best_params)
    data_dict['Heff'].append(Heff)
    data_dict['a16'].append(corner.quantile(flat_samples[:,0], [0.16]))
    data_dict['a50'].append(corner.quantile(flat_samples[:,0], [0.5]))
    data_dict['a84'].append(corner.quantile(flat_samples[:,0], [0.84]))
    data_dict['b16'].append(corner.quantile(flat_samples[:,1], [0.16]))
    data_dict['b50'].append(corner.quantile(flat_samples[:,1], [0.5]))
    data_dict['b84'].append(corner.quantile(flat_samples[:,1], [0.84]))
    data_dict['k16'].append(corner.quantile(flat_samples[:,2], [0.16]))
    data_dict['k50'].append(corner.quantile(flat_samples[:,2], [0.5]))
    data_dict['k84'].append(corner.quantile(flat_samples[:,2], [0.84]))
    data_dict['p16'].append(corner.quantile(flat_samples[:,3], [0.16]))
    data_dict['p50'].append(corner.quantile(flat_samples[:,3], [0.5]))
    data_dict['p84'].append(corner.quantile(flat_samples[:,3], [0.84]))
    data_dict['Z_re'].append(zre_mean)


    # print(corner.quantile(flat_samples[:,0], [0.16, 0.5, 0.84]))
    # print(corner.quantile(flat_samples[:,1], [0.16, 0.5, 0.84]))
    # print(corner.quantile(flat_samples[:,2], [0.16, 0.5, 0.84]))
    # print(corner.quantile(flat_samples[:,3], [0.16, 0.5, 0.84]))

    print(data_dict)
    x0 = np.linspace(0.06, 10, 1000)
    y_plot_fit = sa.lin_bias(kvalues, 0.564,0.593,0.185)


    f, ax = plt.subplots(figsize=(6,4))
    for ind in inds:
        sample = flat_samples[ind]
        ax.plot(x0, (sample[1]/(1+(x0/sample[2]))**sample[0]), alpha=0.05, color='red')
    ax.errorbar(kbins_zre, b_mz, yerr = (sample[3]/((0.2*(kbins_zre/0.3)+1)**(0.6))) , linestyle = 'None',capsize=4, marker ='o') #yerr = bmz_errors*sample[2]
    #plt.plot(kvalues,y_plot_fit, label = 'Battaglia fit') #this lines plots the Battaglia best fit value
    ax.set_xlim(0, 10.)
    ax.set_ylabel(r'$b_{mz}$ ')
    ax.set_xlabel(r'$k[Mpc⁻1 h]$')
    plt.title(r'$b_{zm}$ as a function of k ')
    plt.legend()
    plt.loglog()
    #uncomment the following line for plotting
    #plt.show()

    # f, ax = plt.subplots(figsize=(6,4))
    # for ind in inds:
    #     sample = flat_samples[ind]
    #     ax.plot(x0, (0.593/(1+(x0/sample[1]))**sample[0]), alpha=0.05, color='red')
    # ax.errorbar(kvalues, b_mz, yerr = (sample[2]/(-np.exp((0.7*kvalues)+0.1)+2.2)) , linestyle = 'None',capsize=4, marker ='o') #yerr = bmz_errors*sample[2]
    # plt.plot(kvalues,y_plot_fit, label = 'Battaglia fit') #this lines plots the Battaglia best fit value
    # ax.set_ylabel(r'$b_{mz}$ ')
    # ax.set_xlabel(r'$k[Mpc⁻1 h]$')
    # plt.title(r'$b_{zm}$ as a function of k ')
    # plt.legend()
    # plt.loglog()
    # plt.show()
    """
    f, ax = plt.subplots(figsize=(6,4))
    for ind in inds:
        sample = flat_samples[ind]
        ax.plot(x0, (1./(1+(x0/sample[1]))**sample[0]), alpha=0.05, color='red')
    ax.errorbar(kvalues, b_mz, yerr = bmz_errors, linestyle = 'None',capsize=4, marker ='o')
    #ax.plot(kvalues,y_plot_fit, color = 'b', label = 'values obtain by Battaglia et al. model')
    #ax.set_xlim(0, 10.)
    ax.set_ylabel(r'$b_{mz}$ ')
    ax.set_xlabel(r'$k[Mpc^{⁻1} h]$')
    plt.title(r'$b_{zm}$ as a function of k ')
    plt.legend()
    plt.show()
    """


    sample = flat_samples[50]
    # plt.scatter(kvalues, (b_mz-(1./(1+(kvalues/sample[1]))**sample[0])), color='red')
    # plt.axhline()
    # plt.show()
    #these lines represents curve fitting with scipy's weird algorithm

    """
    # print(freqs[25:], division[25:])
    a1, b = sa.get_param_value(kvalues, b_mz)
    a0,b0,k0 = a1[0:]
    print(a0, b0, k0)
    y_plot_fit = sa.lin_bias(kvalues, a0,b0,k0)
    
    fig, ax = plt.subplots()
    plt.plot(kvalues, y_plot_fit)
    plt.errorbar(kvalues, b_mz, yerr = errs, label = 'data fitting for',linestyle = 'None',capsize=4, marker ='o')
    plt.title(r'best curve fitting for the linear bias')
    plt.show()
    """
print(data_dict)