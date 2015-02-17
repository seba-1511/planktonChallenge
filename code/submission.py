# -*- coding: utf-8 -*-
import os
import warnings
import theano
import inspect
import numpy as np
import pylearn2

from os.path import exists
from skimage.transform import resize
from skimage.io import imread
from itertools import izip_longest

from pylearn2.space import VectorSpace
from pdb import set_trace as d
import cPickle as pk

TEST_PATH = ''

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks, from itertools recipes"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)

def load_test_data(size=28, predict=None, model=None, vec=None, sub=None):
    curr_dir = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe())))
    folders = os.walk(os.path.join(curr_dir, '../data/test/'))
    images = []
    names = []
    print 'Reading test data...'
    for class_id, folder in enumerate(folders):
        for img in folder[2]:
            if img.index('.jpg') == -1:
                continue
            image = imread(folder[0] + '/' + img)
            image = resize(image, (size, size))
            # images.extend(image.ravel())
            # names.append(img)
            p = np.array([image.ravel(), ])
            p = vec.np_format_as((p - 0.5), model.get_input_space())
            p = predict(p)[0]
            l = str(img) + ',' + \
                        ','.join([str(format(i, 'f')) for i in p]) + '\n'
            sub.write(l)
            # print 'Wrote image', img
    # import gzip
    # f = gzip.open('submit.pkl.gz', 'wb')
    # pk.dump((names, images), f, protocol=pk.HIGHEST_PROTOCOL)
    # f.close()
    # f = gzip.open('submit.pkl.gz', 'rb')
    # names, images = pk.load(f)
    # f.close()
    print 'Done reading'
    return np.array(names), np.array(images)


def submit(predict, model=None, img_size=28):
    vec = VectorSpace(img_size**2)
    X = model.get_input_space().make_theano_batch()
    Y = model.fprop(X)
    predict = theano.function([X], Y)
    with open('submission.csv', 'wb') as sub:
        sub.write(
                'image,acantharia_protist_big_center,acantharia_protist_halo,acantharia_protist,amphipods,appendicularian_fritillaridae,appendicularian_s_shape,appendicularian_slight_curve,appendicularian_straight,artifacts_edge,artifacts,chaetognath_non_sagitta,chaetognath_other,chaetognath_sagitta,chordate_type1,copepod_calanoid_eggs,copepod_calanoid_eucalanus,copepod_calanoid_flatheads,copepod_calanoid_frillyAntennae,copepod_calanoid_large_side_antennatucked,copepod_calanoid_large,copepod_calanoid_octomoms,copepod_calanoid_small_longantennae,copepod_calanoid,copepod_cyclopoid_copilia,copepod_cyclopoid_oithona_eggs,copepod_cyclopoid_oithona,copepod_other,crustacean_other,ctenophore_cestid,ctenophore_cydippid_no_tentacles,ctenophore_cydippid_tentacles,ctenophore_lobate,decapods,detritus_blob,detritus_filamentous,detritus_other,diatom_chain_string,diatom_chain_tube,echinoderm_larva_pluteus_brittlestar,echinoderm_larva_pluteus_early,echinoderm_larva_pluteus_typeC,echinoderm_larva_pluteus_urchin,echinoderm_larva_seastar_bipinnaria,echinoderm_larva_seastar_brachiolaria,echinoderm_seacucumber_auricularia_larva,echinopluteus,ephyra,euphausiids_young,euphausiids,fecal_pellet,fish_larvae_deep_body,fish_larvae_leptocephali,fish_larvae_medium_body,fish_larvae_myctophids,fish_larvae_thin_body,fish_larvae_very_thin_body,heteropod,hydromedusae_aglaura,hydromedusae_bell_and_tentacles,hydromedusae_h15,hydromedusae_haliscera_small_sideview,hydromedusae_haliscera,hydromedusae_liriope,hydromedusae_narco_dark,hydromedusae_narco_young,hydromedusae_narcomedusae,hydromedusae_other,hydromedusae_partial_dark,hydromedusae_shapeA_sideview_small,hydromedusae_shapeA,hydromedusae_shapeB,hydromedusae_sideview_big,hydromedusae_solmaris,hydromedusae_solmundella,hydromedusae_typeD_bell_and_tentacles,hydromedusae_typeD,hydromedusae_typeE,hydromedusae_typeF,invertebrate_larvae_other_A,invertebrate_larvae_other_B,jellies_tentacles,polychaete,protist_dark_center,protist_fuzzy_olive,protist_noctiluca,protist_other,protist_star,pteropod_butterfly,pteropod_theco_dev_seq,pteropod_triangle,radiolarian_chain,radiolarian_colony,shrimp_caridean,shrimp_sergestidae,shrimp_zoea,shrimp-like_other,siphonophore_calycophoran_abylidae,siphonophore_calycophoran_rocketship_adult,siphonophore_calycophoran_rocketship_young,siphonophore_calycophoran_sphaeronectes_stem,siphonophore_calycophoran_sphaeronectes_young,siphonophore_calycophoran_sphaeronectes,siphonophore_other_parts,siphonophore_partial,siphonophore_physonect_young,siphonophore_physonect,stomatopod,tornaria_acorn_worm_larvae,trichodesmium_bowtie,trichodesmium_multiple,trichodesmium_puff,trichodesmium_tuft,trochophore_larvae,tunicate_doliolid_nurse,tunicate_doliolid,tunicate_partial,tunicate_salp_chains,tunicate_salp,unknown_blobs_and_smudges,unknown_sticks,unknown_unclassified\n')
        # pred = pred[0:len(names)]
        # for name, p in zip(names, data):
        #     p = predict((p - 0.5), model, 1, vec)[0]
        #     l = str(name) + ',' + \
                #             ','.join([str(format(i, 'f')) for i in p]) + '\n'
        #     sub.write(l)
        names, data = load_test_data(img_size, predict, model, vec, sub)


if __name__ == '__main__':
    pred = lambda d, m, b, v: [[0.00826446 for i in xrange(121)] for a in d]
    submit(pred)
