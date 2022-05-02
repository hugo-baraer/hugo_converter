"""
plot_params.py

Author: Hugo Baraer
Affilitation : Cosmic dawwn group at Mcgill University

This module is used for the comparison between z-reion and 21cmFAST, and deals with redshift of reionization fields.
It also includes the general plotting functions for the z-reion parameters variability range over 21cmFAST inputs (with dict or Json data)
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio
from numpy import array
from statistical_analysis import *
from FFT import *
from z_re_field import *

def reionization_history(redshifts, field,  resolution = 143, plot = True):
    '''
    This function computes the reionization history of a given redshift of reionization field.
    :param redshifts: the redshift range to look the history over (1D array)
    :param field: the 3d redshfit of reionization field
    :param resolution: the resolution of the observed field
    :param plot: to plot or not the histories (True default)
    :return: the ionization history at each given redshfit (1D list)
    '''
    ionization_rate = []
    for i in tqdm(redshifts, 'computing the reionization history'):
        new_box = np.zeros((resolution, resolution, resolution))
        new_box[field <= i] = 1
        ionization_rate.append((new_box == 0).sum() / (resolution) ** 3)
    if plot:
        fig, ax = plt.subplots()
        # plt.scatter(redshifts,ionization_rate)
        plt.plot(redshifts, ionization_rate)
        ax.set_xlabel(r'z ')
        ax.set_ylabel(r'$x_i(z)$')
        plt.legend()
        plt.title(r'ionization fraction as function of redshift')
        plt.show()
    return ionization_rate

def ionization_map_gen(redshift,resolution,field, plot = False):
    '''
    This function computes the ionization maps at a given redshift from a given redshfit of reionization field
    :param redshift: the redshift at which to  compute the ionizaiton maps
    :param resolution: the resolution of the fields
    :param field: the redshift of reionization map to substract from
    :param plot: plot if True
    :return:  the 3D ionization map
    '''
    new_box1 = np.zeros((resolution, resolution, resolution))
    new_box1[field <= redshift] = 1
    if plot:
        fig, ax = plt.subplots()
        if resolution % 2:
            position_vec = np.linspace(-int((resolution*(100/143))//2)-1, int(resolution*(100/143)//2), resolution)
        else:
            position_vec = np.linspace(-int((resolution*(100/143))//2), int(resolution*(100/143)//2), resolution)
        X, Y = np.meshgrid(position_vec, position_vec)
        plt.contourf(X, Y, new_box1[int(143 // 2)], cmap='Blues')
        ax.set_xlabel(r'$[\frac{Mpc}{\hbar}]$')
        ax.set_ylabel(r'$[\frac{Mpc}{\hbar}]$')
        plt.show()
    return new_box1


def ionization_map_diff(redshift,resolution,field1,field2, plot = True):
    '''
    This functions computes the ionization map differences and plots it
    :param redshift: the redshift at which to  compute the ionizaiton maps
    :param resolution: the resolution of the fields
    :param field1: the redshift of reionization map to substract from
    :param field2: the redshift of reionization to be substracted
    :param plot: plot if True
    :return:  the 3D field difference
    '''
    #for the first field
    new_box1 = np.zeros((resolution, resolution, resolution))
    new_box1[field1 <= redshift ] = 1
    #for the second field
    new_box2 = np.zeros((resolution, resolution, resolution))
    new_box2[field2 <= redshift ] = 1
    diff_box = new_box1-new_box2
    if plot:
        fig, ax = plt.subplots()
        if resolution % 2:
            position_vec = np.linspace(-int((resolution*(100/143))//2)-1, int(resolution*(100/143)//2), resolution)
        else:
            position_vec = np.linspace(-int((resolution*(100/143))//2), int(resolution*(100/143)//2), resolution)
        X, Y = np.meshgrid(position_vec, position_vec)
        plt.contourf(X,Y,diff_box[int(resolution//2)], cmap =  'RdBu', vmin=-1.0, vmax=1.0)
        ax.set_xlabel(r'$[\frac{Mpc}{\hbar}]$')
        ax.set_ylabel(r'$[\frac{Mpc}{\hbar}]$')
        plt.show()
    return diff_box

def ps_ion_map(map,radius_thick,resolution, delta =1,plot=False):
    '''
    This module computes the power spectrum of an ionization maps
    :param map: the ionization map to compute the power spectrum of
    :param resolution: the resolution of the fields
    :param radius_thick: the thickness of the shells to average the field over (see average k in statistuical tools for more)
    :param delta: (see fft description)
    :return: the power spewctrum as a function of k (1d array)
    '''
    cx = int(resolution // 2)
    Xr, Yr, field_fft, freq_field = compute_fft(map, delta, resolution)
    field_fft = np.square(abs(field_fft))
    freqs = np.fft.fftshift(np.fft.fftfreq(143, d=1))
    kvalues = np.linspace(0, np.sqrt(3 * (freq_field) ** 2), num=int(np.sqrt(3 * (cx) ** 2) / int(radius_thick)))
    kvalues = kvalues[1:]
    #compute the  average k values in Fourier space to compute the power spectrum
    values, count= average_overk(resolution, field_fft, radius_thick)
    field_fft_k = np.divide(values, count)
    if plot:
        fig, ax = plt.subplots()
        plt.scatter(kvalues, field_fft_k)
        ax.set_xlabel(r'$ k [\frac{\hbar}{Mpc}]$')
        ax.set_ylabel(r'Power spectrum')
        plt.show()
    return field_fft_k


zreionmod = np.load('zreion_for_Hugo.npz')





#b = np.load('zreion_for_Hugo.npy')
a= np.load('zre.npy')
#over_zre, zre_mean = over_zre_field(b[0])



# print(b[0].min())
# print(b[0].max())
#redshifts = np.linspace(6.85,7.95,200)
#redshifts = np.linspace(6,9,4)
#redshifts = [3,4,5,5.5,6,6.2,6.4,6.6,6.8,7.0,7.05, 7.1,7.15,7.2,7.25, 7.3,7.35,7.4,7.45,7.5,7.55,7.6,7.65,7.7,7.75,7.8,7.85, 7.9,7.95,8.0,8.05,8.1,8.15,8.2,8.25,8.3,8.35,8.4,8.45,8.5,8.55,8.6,8.65,8.7,8.75,8.8,8.85,8.9,8.95,9.0,9.05,9.1,9.15,9.2,9.25,9.3,9.35,9.4,9.45,9.5,9.55,9.6,9.65,9.7,9.75,9.8,9.85,9.9,9.95,10.0,10.1,10.2,10.3,10.4,10.5,10.6,10.7,10.8,10.9,11.0,11.1,11.2,11.3,11.4,11.5,11.6,11.7,11.8,11.9,12.0,12.2,12.4,12.6,12.8,13,13.25,13.5,13.75,14,14.25,14.5,14.75,15,15.5,16.5,17,17.5,18,18.5,19,19.5,20,20.5]
redshifts = np.linspace(3.8,16,1000)



print(np.mean(a))
print(np.median(a))



aaa = zreionmod['zreion']
#redshifts = np.linspace(5.5,15,1000)
redshifts = np.linspace(8.0486,16,1)
filenames = []

for i in tqdm(redshifts):

    Xd, Yd, overzre_fft, freqzre = compute_fft((new_box-new_box2), 1, 143)
    cmFastzre_fft = np.square(abs(cmFastzre_fft))
    overzre_fft = np.square(abs(overzre_fft))
    freqs = np.fft.fftshift(np.fft.fftfreq(143, d=1))

    kvalues = np.linspace(0, np.sqrt(3 * (freqzre) ** 2), num=int(np.sqrt(3 * (cx) ** 2) / int(radius_thick)))
    kvalues = kvalues[1:]


    position_vec = np.linspace(-49, 50, 143)
    Xd, Yd = np.meshgrid(position_vec, position_vec)
    # xompute the average for the field
    value_cm, count_cm = average_overk(143, cmFastzre_fft , radius_thick)
    values_zre, count_zre = average_overk(143, overzre_fft,  radius_thick)
    #plt.contourf(overzre_fft[int(143 // 2)])
    fig, ax = plt.subplots()
    #plt.contourf(Xd,Yd,new_box[int(143 // 2)]-new_box2[int(143 // 2)], cmap =  'RdBu', vmin=-1.0, vmax=1.0)

    overzre_fft_k = np.divide(values_zre, count_zre)
    cmFast_fft_k = np.divide(value_cm, count_cm)

    #plt.title(r'slice of the ionization field at a redshift of {} '.format(i))
    # plt.savefig('./ionization_map/ionization_field21cm_{}.png'.format(i))
    # filenames.append('./ionization_map/ionization_field21cm_{}.png'.format(i))
    # plt.savefig('./ionization_map/ionization_field{}.png'.format(i))
    # filenames.append('./ionization_map/ionization_field{}.png'.format(i))
    #plt.colorbar()
    plt.show()
    #plt.close()

#print(zre_mean)

# images = []
# for filename in filenames:
#     images.append(imageio.imread(filename))
# imageio.mimsave('movie_ionization_21cm.gif', images)
#imageio.mimsave('movie_ionization_zreion.gif', images)


#print(ionization_rate)



# fig, ax = plt.subplots()
# plt.contourf(b[1][:][:][75])
# plt.colorbar()
# ax.set_xlabel(r'[Mpc]')
# ax.set_ylabel(r'[Mpc]')
# plt.title(r'slice of a the over-redshift of reionization at the center with a pixel resolution of {} Mpc h⁻¹'.format('1'))


fig, ax = plt.subplots()
plt.contourf(b[2][:][:][75])
plt.colorbar()
ax.set_xlabel(r'[Mpc h⁻¹]')
ax.set_ylabel(r'[Mpc h⁻¹]')
plt.title(r'slice of a the over-redshift of reionization at the center with a pixel resolution of {} Mpc h⁻¹'.format('1'))
plt.show()


a = {'Z_re': [9.396701054337125, 9.283932019860426, 9.144286981051614, 8.972942749948961, 8.826463379644464, 8.671627213805316, 8.516525676875816, 8.312715207917908, 8.141544015180868, 7.9694956615588435, 7.79832344290264, 7.588524683786066, 7.403577790491576, 7.2403290875098785, 7.058309483562552], 'Heff': [7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9], 'medians': [array([0.91816496, 1.12202245, 0.04815797, 0.01374211]), array([0.96192428, 1.04661435, 0.05935293, 0.01247621]), array([0.89825585, 1.0765973 , 0.05143589, 0.01438904]), array([1.03012121, 1.02405853, 0.06881685, 0.015596  ]), array([0.88629944, 1.13438821, 0.04543851, 0.01305808]), array([0.95900476, 1.11530534, 0.0534367 , 0.016161  ]), array([0.90162736, 1.08473575, 0.05031027, 0.01306996]), array([0.88414362, 1.15229063, 0.04379291, 0.01467261]), array([0.88213479, 1.14561677, 0.04411855, 0.01329872]), array([0.82354619, 1.21576482, 0.03460948, 0.01180405]), array([0.83382018, 1.20642103, 0.03633014, 0.01411022]), array([0.90223026, 1.15273858, 0.04442337, 0.01663044]), array([0.84768529, 1.21038519, 0.03609344, 0.01420787]), array([0.80365195, 1.24236022, 0.03150117, 0.01444418]), array([0.74846812, 1.35424301, 0.02439833, 0.01742645])], 'a16': [array([0.91919148]), array([0.89086697]), array([0.88277788]), array([0.88939004]), array([0.86152989]), array([0.8569324]), array([0.82876063]), array([0.83994386]), array([0.81199034]), array([0.81207846]), array([0.79190236]), array([0.7939233]), array([0.79852566]), array([0.7708031]), array([0.7537857])], 'a50': [array([0.96659641]), array([0.9369144]), array([0.9295798]), array([0.94097457]), array([0.91115179]), array([0.89929847]), array([0.86900772]), array([0.87923843]), array([0.85046268]), array([0.85020293]), array([0.83164505]), array([0.83376142]), array([0.83589763]), array([0.80801086]), array([0.79048197])], 'a84': [array([1.01805373]), array([0.98676431]), array([0.97927361]), array([0.99389579]), array([0.96332351]), array([0.94512214]), array([0.91197209]), array([0.92094286]), array([0.89156996]), array([0.89206759]), array([0.87489933]), array([0.87766681]), array([0.87701201]), array([0.84734869]), array([0.8306958])], 'b16': [array([1.03991607]), array([1.04495614]), array([1.05272738]), array([1.05604692]), array([1.06221267]), array([1.07511596]), array([1.09533229]), array([1.11195655]), array([1.12533558]), array([1.12087095]), array([1.13191103]), array([1.15798943]), array([1.17517767]), array([1.19813574]), array([1.23811022])], 'b50': [array([1.06657393]), array([1.07313059]), array([1.08224489]), array([1.0865506]), array([1.09547679]), array([1.10852238]), array([1.13177861]), array([1.14905534]), array([1.16645824]), array([1.16344063]), array([1.17921667]), array([1.20946734]), array([1.22794956]), array([1.25895816]), array([1.30834604])], 'b84': [array([1.0963547]), array([1.10635709]), array([1.1174497]), array([1.12454827]), array([1.13743487]), array([1.14694078]), array([1.17380573]), array([1.18916762]), array([1.2128002]), array([1.21239595]), array([1.23508594]), array([1.26979179]), array([1.28977907]), array([1.33054567]), array([1.39319373])], 'k16': [array([0.05103315]), array([0.0481762]), array([0.04703823]), array([0.0471818]), array([0.04391929]), array([0.04251908]), array([0.03917322]), array([0.03900833]), array([0.03566063]), array([0.03472404]), array([0.03235533]), array([0.03092383]), array([0.02957774]), array([0.026396]), array([0.02380858])], 'k50': [array([0.0575195]), array([0.0544741]), array([0.05358098]), array([0.05418379]), array([0.0507811]), array([0.04819353]), array([0.04451395]), array([0.04409987]), array([0.04042883]), array([0.03959684]), array([0.03740933]), array([0.03579055]), array([0.03405229]), array([0.03070715]), array([0.02795813])], 'k84': [array([0.06481614]), array([0.06158762]), array([0.0607405]), array([0.06152289]), array([0.05813161]), array([0.05461512]), array([0.05042011]), array([0.04968332]), array([0.04596644]), array([0.04517002]), array([0.04290505]), array([0.04129801]), array([0.03907767]), array([0.03543515]), array([0.03258164])], 'p16': [array([0.01234072]), array([0.01242572]), array([0.01264166]), array([0.01301108]), array([0.01326862]), array([0.01270141]), array([0.01260699]), array([0.01236695]), array([0.01274681]), array([0.01296933]), array([0.01356192]), array([0.0140807]), array([0.01361917]), array([0.01368417]), array([0.01412747])], 'p50': [array([0.01352539]), array([0.01359935]), array([0.01389093]), array([0.01427796]), array([0.01456819]), array([0.01391331]), array([0.01381404]), array([0.01353384]), array([0.0139517]), array([0.01421298]), array([0.01487084]), array([0.01543753]), array([0.01494865]), array([0.0150288]), array([0.01546528])], 'p84': [array([0.01487546]), array([0.0149929]), array([0.01527847]), array([0.01595142]), array([0.01615175]), array([0.01529999]), array([0.0151997]), array([0.01493231]), array([0.01535871]), array([0.0156907]), array([0.01636462]), array([0.01698239]), array([0.01645685]), array([0.01658366]), array([0.01703918])]}
b = {'Z_re': [2.4447038120078366, 4.189303630009777, 5.002679358882596, 5.5369602767519535, 5.974575329311502, 6.314296833295318, 6.585703406085821, 6.80900462928924, 7.030162365386582, 7.2209361375579775, 7.372540316058337, 7.524227935984012, 7.648473244199197, 7.767343419942569, 7.8799575406255435, 7.991388434539689, 8.09683719381015, 8.195645862279928, 8.281522135744837, 8.360399930647864, 8.446752572577797, 8.51515436492697, 8.580785149614922, 8.643948598714113, 8.704875886009438, 8.765059039938008, 8.822138788396307, 8.877833887956632, 8.932503068353233, 8.986066307891337, 9.038525316436218, 9.092207220624259], 'Heff': [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96], 'medians': [array([0.67820421, 2.37819193, 0.02012686, 0.0305391 ]), array([0.58069955, 2.28707347, 0.01294444, 0.02443398]), array([0.7089804 , 1.68648271, 0.02335719, 0.02060211]), array([0.71296127, 1.75789146, 0.02120249, 0.01832955]), array([0.74899997, 1.53267205, 0.02775316, 0.02120867]), array([0.83401622, 1.50688849, 0.03152588, 0.01659543]), array([0.76795947, 1.68456615, 0.02306855, 0.02179632]), array([0.89880495, 1.38409356, 0.0371032 , 0.02339746]), array([0.82578002, 1.59057517, 0.02597772, 0.02132738]), array([0.85549434, 1.52185894, 0.02988826, 0.02030037]), array([0.82328785, 1.62216188, 0.02499712, 0.02229914]), array([0.87259237, 1.48032297, 0.03108958, 0.01854296]), array([0.94069335, 1.45764487, 0.03387513, 0.02101927]), array([0.99588922, 1.33268293, 0.04124865, 0.02399893]), array([0.83872133, 1.63173393, 0.02473375, 0.02271535]), array([0.87470033, 1.65745566, 0.0254436 , 0.0221859 ]), array([0.94303178, 1.50470651, 0.03107342, 0.01801905]), array([0.91223206, 1.56925813, 0.02757401, 0.02272781]), array([0.77361904, 1.83765661, 0.01712159, 0.02227504]), array([0.95217855, 1.52761705, 0.03108174, 0.02340588]), array([0.96617067, 1.35456332, 0.03910781, 0.0246104 ]), array([0.93595512, 1.44903951, 0.03242727, 0.02427982]), array([0.85085426, 1.68032536, 0.02229842, 0.02465398]), array([0.94408139, 1.4835386 , 0.03037015, 0.02437015]), array([1.04913846, 1.31080257, 0.04185847, 0.02147918]), array([0.99888562, 1.33270872, 0.03782382, 0.02611196]), array([0.92563991, 1.44408361, 0.03109314, 0.02604089]), array([1.03405673, 1.35013779, 0.03906027, 0.02422579]), array([1.03948884, 1.42329563, 0.03488679, 0.02351002]), array([0.86478725, 1.53721433, 0.02420433, 0.02502088]), array([0.95885833, 1.47910244, 0.02980264, 0.0243835 ]), array([0.97282858, 1.61009335, 0.02586306, 0.02295738])], 'a16': [array([0.66456685]), array([0.65217863]), array([0.67854491]), array([0.71924908]), array([0.71957893]), array([0.73841017]), array([0.74471917]), array([0.77706338]), array([0.78240452]), array([0.78546619]), array([0.80966418]), array([0.79988536]), array([0.82445022]), array([0.84033427]), array([0.85021226]), array([0.85311799]), array([0.86139726]), array([0.86263545]), array([0.87088038]), array([0.88156714]), array([0.8651304]), array([0.87719979]), array([0.8881233]), array([0.90308319]), array([0.91438787]), array([0.91935462]), array([0.92905955]), array([0.93812791]), array([0.93952633]), array([0.94465057]), array([0.95075546]), array([0.94568216])], 'a50': [array([0.68604164]), array([0.68251861]), array([0.7111957]), array([0.75227322]), array([0.75606103]), array([0.77389378]), array([0.78308992]), array([0.81877228]), array([0.82464559]), array([0.82967838]), array([0.85550275]), array([0.84457222]), array([0.87331754]), array([0.88955689]), array([0.90182873]), array([0.90595195]), array([0.91383359]), array([0.91594424]), array([0.92465106]), array([0.93584446]), array([0.92166764]), array([0.93216544]), array([0.94589976]), array([0.96494596]), array([0.97545805]), array([0.98149364]), array([0.99202282]), array([1.00311339]), array([1.00137395]), array([1.01424743]), array([1.01720518]), array([1.01152353])], 'a84': [array([0.71115326]), array([0.71470369]), array([0.74624116]), array([0.78887306]), array([0.79579962]), array([0.81208631]), array([0.82490567]), array([0.86420134]), array([0.87069962]), array([0.87549417]), array([0.9060375]), array([0.8944769]), array([0.92651992]), array([0.94444407]), array([0.96182893]), array([0.96240352]), array([0.97264259]), array([0.97884204]), array([0.9857719]), array([0.99602891]), array([0.98417591]), array([0.99439594]), array([1.01369222]), array([1.03232973]), array([1.0422521]), array([1.05186571]), array([1.06349683]), array([1.07527168]), array([1.07424538]), array([1.09156667]), array([1.08898148]), array([1.0850625])], 'b16': [array([2.251937]), array([1.51136294]), array([1.51655134]), array([1.4942101]), array([1.49323313]), array([1.49262669]), array([1.49446428]), array([1.45785069]), array([1.45422551]), array([1.45729317]), array([1.43435563]), array([1.4510488]), array([1.41906328]), array([1.40329616]), array([1.39489088]), array([1.4052801]), array([1.39686411]), array([1.38779146]), array([1.38559639]), array([1.38505979]), array([1.39266576]), array([1.38562533]), array([1.36772176]), array([1.35050161]), array([1.34825746]), array([1.34365805]), array([1.3374548]), array([1.33737173]), array([1.34696735]), array([1.33539594]), array([1.34203414]), array([1.34442955])], 'b50': [array([2.39212889]), array([1.6219032]), array([1.62646174]), array([1.59425206]), array([1.59954522]), array([1.59372848]), array([1.60393091]), array([1.55933281]), array([1.55713965]), array([1.55749721]), array([1.53176744]), array([1.55539013]), array([1.51653562]), array([1.49828506]), array([1.4941061]), array([1.500155]), array([1.49170209]), array([1.48871588]), array([1.48088367]), array([1.47526997]), array([1.48840028]), array([1.47876501]), array([1.46332957]), array([1.44198567]), array([1.43559926]), array([1.43065616]), array([1.42785534]), array([1.42387726]), array([1.43469826]), array([1.42464794]), array([1.42631556]), array([1.43630206])], 'b84': [array([2.47072812]), array([1.75205421]), array([1.76291059]), array([1.71461701]), array([1.73375291]), array([1.71922766]), array([1.74041174]), array([1.68139007]), array([1.68046532]), array([1.68802655]), array([1.65406403]), array([1.68009527]), array([1.64161795]), array([1.61271215]), array([1.606767]), array([1.61837556]), array([1.60694809]), array([1.60953675]), array([1.59839816]), array([1.58887716]), array([1.60840825]), array([1.59626395]), array([1.58281501]), array([1.55435924]), array([1.54185372]), array([1.53818041]), array([1.53254678]), array([1.52740377]), array([1.53756295]), array([1.53531466]), array([1.53377329]), array([1.54551817])], 'k16': [array([0.01858722]), array([0.02123097]), array([0.02116376]), array([0.02249961]), array([0.02179124]), array([0.02191744]), array([0.02169343]), array([0.02343203]), array([0.02317008]), array([0.02270396]), array([0.02381445]), array([0.02307182]), array([0.02438148]), array([0.02529267]), array([0.02554101]), array([0.02512719]), array([0.02536482]), array([0.02513464]), array([0.02539681]), array([0.02588435]), array([0.02502347]), array([0.0255083]), array([0.02600641]), array([0.02712867]), array([0.02764913]), array([0.02778849]), array([0.02806067]), array([0.02845896]), array([0.02805715]), array([0.02818482]), array([0.0282588]), array([0.02771777])], 'k50': [array([0.01982712]), array([0.02507559]), array([0.02513814]), array([0.02637011]), array([0.02592932]), array([0.02585692]), array([0.02586292]), array([0.02779916]), array([0.02758706]), array([0.02726481]), array([0.02854061]), array([0.02753327]), array([0.02931464]), array([0.03022707]), array([0.0304883]), array([0.03021144]), array([0.03055058]), array([0.03035443]), array([0.03063239]), array([0.03115998]), array([0.03029305]), array([0.03077314]), array([0.03150902]), array([0.03284666]), array([0.03328086]), array([0.03361055]), array([0.03392153]), array([0.03434739]), array([0.03368032]), array([0.03440212]), array([0.03427589]), array([0.03356978])], 'k84': [array([0.02210726]), array([0.02933071]), array([0.02956638]), array([0.0307745]), array([0.0305573]), array([0.03019742]), array([0.03063087]), array([0.03283014]), array([0.0326061]), array([0.0320595]), array([0.03376005]), array([0.03282218]), array([0.03493163]), array([0.03600226]), array([0.03673264]), array([0.03600242]), array([0.03631743]), array([0.03676325]), array([0.03681262]), array([0.03704161]), array([0.03647049]), array([0.03679229]), array([0.03828141]), array([0.03954995]), array([0.03989391]), array([0.04029167]), array([0.04102358]), array([0.04117345]), array([0.04050079]), array([0.04184082]), array([0.04105286]), array([0.04046513])], 'p16': [array([0.02828658]), array([0.01638152]), array([0.0169032]), array([0.01645835]), array([0.01728926]), array([0.01684764]), array([0.01802863]), array([0.01815319]), array([0.01810147]), array([0.0180891]), array([0.0183444]), array([0.01918629]), array([0.01939671]), array([0.01937719]), array([0.0193702]), array([0.01930242]), array([0.01927463]), array([0.0194303]), array([0.01967277]), array([0.01985863]), array([0.02060795]), array([0.02069283]), array([0.020702]), array([0.02078111]), array([0.02074345]), array([0.02081746]), array([0.02090737]), array([0.02083131]), array([0.02088194]), array([0.02098735]), array([0.02099244]), array([0.02115601])], 'p50': [array([0.03102903]), array([0.01792838]), array([0.01846805]), array([0.01805508]), array([0.0189315]), array([0.01849991]), array([0.01975651]), array([0.01990947]), array([0.01982792]), array([0.01985266]), array([0.02007009]), array([0.02102076]), array([0.02119819]), array([0.02116727]), array([0.02122989]), array([0.02114211]), array([0.02111985]), array([0.02129444]), array([0.02149811]), array([0.0217764]), array([0.02253839]), array([0.02267288]), array([0.02279089]), array([0.02277871]), array([0.02272454]), array([0.02281119]), array([0.02290772]), array([0.02288628]), array([0.02292312]), array([0.02303319]), array([0.02301076]), array([0.0230957])], 'p84': [array([0.03433467]), array([0.01973708]), array([0.02034006]), array([0.01992919]), array([0.02087723]), array([0.02035499]), array([0.0217374]), array([0.02195341]), array([0.0218862]), array([0.02182405]), array([0.02207152]), array([0.02317768]), array([0.02334819]), array([0.02333737]), array([0.02350985]), array([0.02327419]), array([0.02324945]), array([0.02369203]), array([0.0236243]), array([0.02394874]), array([0.0247922]), array([0.02493012]), array([0.02512827]), array([0.02515452]), array([0.02497634]), array([0.02514934]), array([0.02524154]), array([0.02523403]), array([0.02524184]), array([0.02545687]), array([0.025385]), array([0.02542237])]}
#a = {'Z_re': [7.2209361375579775, 7.2209361375579775, 7.2209361375579775, 7.2209361375579775, 7.2209361375579775, 7.2209361375579775, 7.2209361375579775, 7.2209361375579775, 7.2209361375579775, 7.2209361375579775, 7.2209361375579775, 7.2209361375579775, 7.2209361375579775, 7.2209361375579775, 7.2209361375579775, 7.2209361375579775, 7.2209361375579775, 7.2209361375579775, 7.2209361375579775, 7.2209361375579775, 7.2209361375579775, 7.2209361375579775], 'Heff': [1.1, 1.6, 2.1, 2.6, 3.1, 3.6, 4.1, 4.6, 5.1, 5.6, 6.1, 6.6, 7.1, 7.6, 8.1, 8.6, 9.1, 9.6, 10.1, 10.6, 11.1, 11.6], 'medians': [array([0.90577744, 1.46361324, 0.03400081, 0.02445637]), array([0.84589188, 1.46843829, 0.03006966, 0.0196655 ]), array([0.86328173, 1.58771308, 0.02791629, 0.02191205]), array([0.77242055, 1.88436911, 0.01900421, 0.01771325]), array([0.87626126, 1.42186702, 0.03381752, 0.01938556]), array([0.79895951, 1.69006035, 0.02322239, 0.01735379]), array([0.76787728, 1.64309474, 0.02260908, 0.01783903]), array([0.8446086 , 1.6343743 , 0.02536155, 0.02247602]), array([0.89343599, 1.46177181, 0.03217414, 0.01994617]), array([0.84227957, 1.44162088, 0.03079672, 0.02116333]), array([0.86058937, 1.55240226, 0.02860275, 0.02057329]), array([0.8737018 , 1.46578456, 0.0314661 , 0.0192532 ]), array([0.92437935, 1.42726688, 0.03650221, 0.01792166]), array([0.84609425, 1.46051299, 0.03075909, 0.02260688]), array([0.76941743, 1.66705011, 0.02319721, 0.01908637]), array([0.84062861, 1.51189147, 0.02951573, 0.01904201]), array([0.81109074, 1.58763506, 0.02488802, 0.02162532]), array([0.80305416, 1.60069467, 0.02503243, 0.01724789]), array([0.80864872, 1.53041336, 0.02802679, 0.02140059]), array([0.81839519, 1.6062633 , 0.02566687, 0.01557648]), array([0.81356022, 1.62300852, 0.0251947 , 0.02206215]), array([0.83238434, 1.62779561, 0.0260049 , 0.0208798 ])], 'a16': [array([0.78806552]), array([0.78517328]), array([0.7854664]), array([0.78696148]), array([0.78686242]), array([0.78439029]), array([0.7874782]), array([0.78551178]), array([0.78456446]), array([0.78737868]), array([0.78720355]), array([0.7860208]), array([0.78768189]), array([0.78583674]), array([0.78766201]), array([0.78788691]), array([0.78675664]), array([0.78607521]), array([0.78880438]), array([0.78647041]), array([0.78796177]), array([0.78647229])], 'a50': [array([0.82863059]), array([0.82904164]), array([0.82890348]), array([0.83000046]), array([0.82866287]), array([0.82809153]), array([0.83039967]), array([0.82759372]), array([0.8269148]), array([0.83087866]), array([0.83027849]), array([0.82848236]), array([0.8306581]), array([0.82807311]), array([0.83040136]), array([0.83044302]), array([0.8301376]), array([0.83063261]), array([0.83085629]), array([0.83051822]), array([0.83107372]), array([0.82864749])], 'a84': [array([0.87460461]), array([0.87615896]), array([0.87476956]), array([0.87784299]), array([0.87579882]), array([0.87643227]), array([0.87775166]), array([0.87388199]), array([0.87323761]), array([0.87751193]), array([0.87585025]), array([0.87423816]), array([0.87895712]), array([0.87300324]), array([0.87634031]), array([0.8776241]), array([0.8773073]), array([0.8774747]), array([0.87783431]), array([0.87746118]), array([0.87871312]), array([0.87607078])], 'b16': [array([1.45664624]), array([1.45292464]), array([1.45494098]), array([1.45176356]), array([1.45313932]), array([1.45618304]), array([1.45208838]), array([1.45908739]), array([1.45986861]), array([1.45198945]), array([1.45347192]), array([1.45666619]), array([1.4495273]), array([1.46008698]), array([1.4549216]), array([1.45044825]), array([1.45236282]), array([1.4509366]), array([1.45157011]), array([1.45210149]), array([1.44904135]), array([1.45320323])], 'b50': [array([1.55793839]), array([1.55456703]), array([1.55985481]), array([1.55642247]), array([1.55654668]), array([1.5625737]), array([1.55280473]), array([1.56090171]), array([1.56202061]), array([1.55234793]), array([1.55493837]), array([1.56063798]), array([1.55409866]), array([1.55841805]), array([1.55254745]), array([1.55155511]), array([1.55393013]), array([1.55594917]), array([1.55502297]), array([1.55464781]), array([1.55234166]), array([1.55782968])], 'b84': [array([1.68144365]), array([1.68503519]), array([1.68807615]), array([1.68187747]), array([1.68102738]), array([1.69039653]), array([1.67912029]), array([1.6830018]), array([1.69239045]), array([1.67796593]), array([1.68028786]), array([1.68472858]), array([1.68233237]), array([1.68368358]), array([1.67880387]), array([1.67692048]), array([1.6800154]), array([1.68592326]), array([1.67650515]), array([1.68453115]), array([1.67436874]), array([1.68224858])], 'k16': [array([0.02289943]), array([0.02270414]), array([0.02267832]), array([0.02283987]), array([0.02287208]), array([0.02261611]), array([0.02290606]), array([0.02271147]), array([0.0225687]), array([0.02293833]), array([0.02289466]), array([0.02282393]), array([0.02287636]), array([0.02275256]), array([0.02289886]), array([0.02296491]), array([0.02284974]), array([0.02274619]), array([0.02300656]), array([0.02279834]), array([0.02301888]), array([0.02284948])], 'k50': [array([0.02717608]), array([0.0272138]), array([0.02716582]), array([0.02727701]), array([0.02722059]), array([0.0270072]), array([0.02739267]), array([0.02705045]), array([0.02701955]), array([0.02738902]), array([0.02734087]), array([0.02710051]), array([0.02732357]), array([0.02714067]), array([0.02736258]), array([0.02744873]), array([0.02731873]), array([0.02730264]), array([0.0274061]), array([0.02737825]), array([0.02745013]), array([0.02720973])], 'k84': [array([0.03207794]), array([0.03230948]), array([0.03216458]), array([0.03240925]), array([0.03229715]), array([0.03230428]), array([0.03238532]), array([0.03195787]), array([0.03198902]), array([0.03248852]), array([0.03233068]), array([0.03210542]), array([0.03260077]), array([0.03191629]), array([0.03222552]), array([0.03250096]), array([0.032421]), array([0.03253983]), array([0.03246571]), array([0.032375]), array([0.03256946]), array([0.03222829])], 'p16': [array([0.01807453]), array([0.0181077]), array([0.01808481]), array([0.01803782]), array([0.018111]), array([0.01803706]), array([0.01809091]), array([0.0180794]), array([0.01809471]), array([0.01807415]), array([0.01808701]), array([0.01804483]), array([0.01812662]), array([0.01810282]), array([0.01812761]), array([0.01807161]), array([0.01809948]), array([0.01806431]), array([0.01801414]), array([0.01808945]), array([0.01808817]), array([0.01812145])], 'p50': [array([0.01975636]), array([0.01982008]), array([0.01984941]), array([0.01975484]), array([0.01985322]), array([0.0197812]), array([0.01986149]), array([0.01977804]), array([0.01983318]), array([0.01979492]), array([0.01982953]), array([0.01981327]), array([0.01987834]), array([0.01981709]), array([0.01987006]), array([0.01978378]), array([0.01982016]), array([0.01980074]), array([0.01978741]), array([0.01982037]), array([0.0197995]), array([0.01987712])], 'p84': [array([0.02183725]), array([0.02181698]), array([0.02189472]), array([0.02177428]), array([0.02183915]), array([0.02181046]), array([0.02183588]), array([0.02180215]), array([0.02185737]), array([0.02182868]), array([0.02182202]), array([0.02182054]), array([0.02193271]), array([0.02184326]), array([0.02192942]), array([0.02189573]), array([0.02188062]), array([0.02185279]), array([0.02184859]), array([0.02194103]), array([0.02181902]), array([0.02189141])]}
print(np.concatenate(a['a50']))
print(np.concatenate(a['a50'])-np.concatenate(a['a16']))


fig3, ax3 = plt.subplots(3,2,sharex='col',sharey='row')

cont300 = ax3[0, 0].errorbar(a['Heff'], np.concatenate(a['a50']), yerr=(np.concatenate(a['a50'])-np.concatenate(a['a16']),np.concatenate(a['a84'])-np.concatenate(a['a50'])), ls = '')
cont300 = ax3[0, 0].scatter(a['Heff'], np.concatenate(a['a50']), color = 'r')
ax3[0, 1].errorbar(b['Heff'], np.concatenate(b['a50']), yerr=(np.concatenate(b['a50'])-np.concatenate(b['a16']),np.concatenate(b['a84'])-np.concatenate(b['a50'])), ls = '')
ax3[0, 1].scatter(b['Heff'], np.concatenate(b['a50']), color = 'r')
ax3[1, 0].errorbar(a['Heff'], np.concatenate(a['b50']), yerr=(np.concatenate(a['b50'])-np.concatenate(a['b16']),np.concatenate(a['b84'])-np.concatenate(a['b50'])), ls = '')
ax3[1, 0].scatter(a['Heff'], np.concatenate(a['b50']), color = 'r')
ax3[1, 1].errorbar(b['Heff'][1:], np.concatenate(b['b50'][1:]), yerr=(np.concatenate(b['b50'][1:])-np.concatenate(b['b16'][1:]),np.concatenate(b['b84'][1:])-np.concatenate(b['b50'][1:])), ls = '')
ax3[1, 1].scatter(b['Heff'][1:], np.concatenate(b['b50'][1:]), color = 'r')
ax3[2, 0].errorbar(a['Heff'], np.concatenate(a['k50']), yerr=(np.concatenate(a['k50'])-np.concatenate(a['k16']),np.concatenate(a['k84'])-np.concatenate(a['k50'])), ls = '')
ax3[2, 0].scatter(a['Heff'], np.concatenate(a['k50']), color = 'r')
ax3[2, 1].errorbar(b['Heff'][1:], np.concatenate(b['k50'][1:]), yerr=(np.concatenate(b['k50'][1:])-np.concatenate(b['k16'][1:]),np.concatenate(b['k84'][1:])-np.concatenate(b['k50'][1:])), ls = '')
ax3[2, 1].scatter(b['Heff'][1:], np.concatenate(b['k50'][1:]), color = 'r')

plt.setp(ax3[2,1], xlabel='Ionization efficiency')
plt.setp(ax3[2,0], xlabel=r'turnover mass $[log_{10}(M_\odot)]$')
plt.setp(ax3[0,0], ylabel= r'$\alpha$')
plt.setp(ax3[1,0], ylabel= r'$b_0$')
plt.setp(ax3[2,0], ylabel= r'$k_0$')


fig, ax = plt.subplots()
plt.errorbar(a['Heff'], np.concatenate(a['a50']), yerr=(np.concatenate(a['a50'])-np.concatenate(a['a16']),np.concatenate(a['a84'])-np.concatenate(a['a50'])), ls = '')
plt.scatter(a['Heff'], np.concatenate(a['a50']), color = 'r')
#plt.xlabel('Ionization efficiency')
plt.xlabel(r'turnover mass $[log_{10}(M_\odot)]$')
plt.ylabel(r'$\alpha$ best fitted value')
plt.show()

fig, ax = plt.subplots()
plt.errorbar(a['Heff'], np.concatenate(a['b50']), yerr=(np.concatenate(a['b50'])-np.concatenate(a['b16']),np.concatenate(a['b84'])-np.concatenate(a['b50'])), ls = '')
plt.scatter(a['Heff'], np.concatenate(a['b50']), color = 'r')
#plt.xlabel('Ionization efficiency')
plt.xlabel(r'turnover mass $[log_{10}(M_\odot)]$')
plt.ylabel(r'$b_0$ best fitted value')
plt.show()

fig, ax = plt.subplots()
plt.errorbar(a['Heff'], np.concatenate(a['k50']), yerr=(np.concatenate(a['k50'])-np.concatenate(a['k16']),np.concatenate(a['k84'])-np.concatenate(a['k50'])), ls = '')
plt.scatter(a['Heff'], np.concatenate(a['k50']), color = 'r')
#plt.xlabel('Ionization efficiency')
plt.xlabel(r'turnover mass $[log_{10}(M_\odot)]$')
plt.ylabel(r'$k_0$ best fitted value')
plt.show()

fig, ax = plt.subplots()
plt.scatter(a['Heff'], a['Z_re'], color = 'r')
#plt.xlabel('Ionization efficiency')
plt.xlabel(r'turnover mass $[log_{10}(M_\odot)]$')
plt.ylabel(r'mean redshift of reionization')
plt.show()

fig, ax = plt.subplots()
plt.errorbar(a['Heff'], np.concatenate(a['a50']), yerr=(np.concatenate(a['a50'])-np.concatenate(a['a16']),np.concatenate(a['a84'])-np.concatenate(a['a50'])), ls = '', label = r'$\alpha$', color = 'r')
plt.errorbar(a['Heff'], np.concatenate(a['b50']), yerr=(np.concatenate(a['b50'])-np.concatenate(a['b16']),np.concatenate(a['b84'])-np.concatenate(a['b50'])), ls = '', label = r'$b_0$', color = 'g')
plt.errorbar(a['Heff'], np.concatenate(a['k50']), yerr=(np.concatenate(a['k50'])-np.concatenate(a['k16']),np.concatenate(a['k84'])-np.concatenate(a['k50'])), ls = '', label = r'$k_0$', color = 'b')
plt.scatter(a['Heff'], np.concatenate(a['k50']), color = 'b')
plt.scatter(a['Heff'], np.concatenate(a['b50']), color = 'g')
plt.scatter(a['Heff'], np.concatenate(a['a50']), color = 'r')
plt.xlabel('Ionization efficiency')
plt.ylabel(r'$\alpha$ best fitted value')
plt.legend()
plt.show()

