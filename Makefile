#
#  Makefile for arasmus' frivolous pursuits.
#

###
#
# RULES
#
###


HEADERS=
BINARIES=	$(SOURCES.c:.c=)
SCRIPTS=	$(SOURCES.sh:.sh=) $(SOURCES.csh:.csh=)
PROGRAMS=	$(BINARIES) $(SCRIPTS)
BINDIR=         .
NUMREC=		./numrec/code/
NUMLIB=		./numrec/lib/
CC=             gcc 
CFLAGS=   	-Wall -g -Icfitsio	
NRFLAGS=	-I./numrec/include
NRF    =        -I./src/recipes_c-ansi/include
# -Bstatic
# LDFLAGS=	-Lcfitsio/lib -lcfitsio -lm 
LDFLAGS=	-Lcfitsio -lcfitsio -lm 
LDF    =	-Lcfitsio -L/usr/local/lib \
		-lcfitsio -lrecipes_c -lm 
FFLAGS=		

###
#
# TARGETS
#
#
###

FAST_2DINT=	fast_2d_interp.o f_2d_tree.o

fast_2d_interp: $(FAST_2DINT)
		$(LINK.c) -o $@ $(FAST_2DINT) $(LDFLAGS)

P4P9=		p4_p9.o $(NUMLIB)ran2.o $(NUMLIB)gasdev.o $(NUMLIB)ran1.o \
		$(NUMLIB)bnldev.o $(NUMLIB)gammln.o $(NUMLIB)nrutil.o

p4_p9:		$(P4P9)
		$(LINK.c) -o $@ $(P4P9) $(LDFLAGS)

LSST_FP_DEMUX=	lsst_fp_demux.o ray_util.o init_ampcoordtran.o

lsst_fp_demux:	$(LSST_FP_DEMUX) ampcoordtran.h
		$(LINK.c) -o $@ $(LSST_FP_DEMUX) $(LDFLAGS)

ray_util.o:	ray_util.c ray.h
		$(COMPILE.c) -o $@ ray_util.c

RAYUTIL=	ray_util.o ray.h $(NUMLIB)factrl.o $(NUMLIB)gammln.o \
		$(NUMLIB)nrutil.o

SOPRA2CPX=	sopra2cpx.o sopra_nk.o 

sopra2cpx:	$(SOPRA2CPX) sopra_nk.h
		$(LINK.c) -o $@ $(SOPRA2CPX) $(LDFLAGS)

DIFFUSER=	diffuser.o $(RAYUTIL) $(NUMLIB)expdev.o $(NUMLIB)ran1.o

diffuser:  	$(DIFFUSER) ray.h
		$(LINK.c) -o $@ $(DIFFUSER) $(LDFLAGS)

IMAGE=		image.o $(RAYUTIL) $(NUMLIB)poidev.o $(NUMLIB)gasdev.o \
		$(NUMLIB)ran1.o

image:  	$(IMAGE) ray.h
		$(LINK.c) -o $@ $(IMAGE) $(LDFLAGS)


SENSOR_IMAGE=	sensor_image.o lsst_fp.o bi_ccd_func.o bi_ccd_Si_funcs.o \
		zernike.o henke_utils.o multilayer.o complex_n_server.o \
		$(RAYUTIL) \
		$(NUMLIB)expdev.o  $(NUMLIB)hunt.o \
		$(NUMLIB)poidev.o $(NUMLIB)gasdev.o $(NUMLIB)ran1.o \
		$(NUMLIB)ran2.o

sensor_image:  	$(SENSOR_IMAGE) ray.h camera_geometry.h complex_n_server.h
		$(LINK.c) -o $@ $(SENSOR_IMAGE) $(LDFLAGS)

CHECK_ALPHA=	bi_ccd_func.o check_alpha.o ray_util.o henke_utils.o \
		$(NUMLIB)expdev.o $(NUMLIB)ran1.o \
		$(NUMLIB)nrutil.o $(NUMLIB)gasdev.o \
		$(NUMLIB)poidev.o $(NUMLIB)gammln.o \
		$(NUMLIB)hunt.o 

check_alpha:	$(CHECK_ALPHA)
		$(LINK.c) -o $@ $(CHECK_ALPHA)

BP_FILTER=	bp_filter.o multilayer.o bi_ccd_Si_funcs.o henke_utils.o \
		$(NUMLIB)hunt.o 

bp_filter:	$(BP_FILTER) bi_ccd.h henke.h ray.h multilayer.h
		$(LINK.c) -o $@ $(BP_FILTER)

multilayer.o:	multilayer.c complex_n_server.c

ML_DESIGN=	ml_design.o multilayer.o complex_n_server.o \
		bi_ccd_func.o bi_ccd_Si_funcs.o henke_utils.o \
		ray_util.o \
		$(NUMLIB)expdev.o $(NUMLIB)gasdev.o \
		$(NUMLIB)poidev.o $(NUMLIB)gammln.o \
		$(NUMLIB)ran1.o $(NUMLIB)ran2.o $(NUMLIB)hunt.o \
		$(NUMLIB)nrutil.o

ml_design:	$(ML_DESIGN) bi_ccd.h henke.h ray.h multilayer.h \
		complex_n_server.h
		$(LINK.c) -o $@ $(ML_DESIGN)

ARCT=		arct.o multilayer.o bi_ccd_Si_funcs.o henke_utils.o \
		$(NUMLIB)hunt.o 

arct:		$(ARCT) bi_ccd.h henke.h ray.h multilayer.h
		$(LINK.c) -o $@ $(ARCT)

ARCOAT=		arcoat.o bi_ccd_Si_funcs.o henke_utils.o $(NUMLIB)hunt.o

arcoat:		$(ARCOAT) bi_ccd.h henke.h ray.h
		$(LINK.c) -o $@ $(ARCOAT)

### SHOULD BI_CCD ALWYAYS KNOW ABOUT MULTILAYER???

BI_CCD_PIXPART=	bi_ccd_pixpart.o bi_ccd_func.o bi_ccd_Si_funcs.o \
		xray_bi_ccd.o field_superpose.o multilayer.o henke_utils.o \
		ray_util.o complex_n_server.o \
		$(NUMLIB)bnldev.o $(NUMLIB)bessk1.o $(NUMLIB)bessi1.o \
		$(NUMLIB)hunt.o $(NUMLIB)gammln.o \
		$(NUMLIB)expdev.o $(NUMLIB)ran1.o $(NUMLIB)ran2.o \
		$(NUMLIB)nrutil.o $(NUMLIB)gasdev.o $(NUMLIB)poidev.o \

bi_ccd_pixpart:	$(BI_CCD_PIXPART) bi_ccd.h ray.h
		$(LINK.c) -o $@ $(BI_CCD_PIXPART)

BI_CCD=		bi_ccd.o bi_ccd_func.o bi_ccd_Si_funcs.o xray_bi_ccd.o \
		ray_util.o henke_utils.o multilayer.o complex_n_server.o \
		field_superpose.o \
		$(NUMLIB)hunt.o \
		$(NUMLIB)nrutil.o $(NUMLIB)ran1.o $(NUMLIB)expdev.o \
		$(NUMLIB)gasdev.o $(NUMLIB)poidev.o $(NUMLIB)gammln.o \
		$(NUMLIB)ran2.o $(NUMLIB)bnldev.o $(NUMLIB)bessk1.o \
		$(NUMLIB)bessi1.o 

bi_ccd: 	$(BI_CCD) ray.h henke.h bi_ccd.h complex_n_server.h \
		field_superpose.h
		$(LINK.c) -o $@ $(BI_CCD)

BI_CCD_DRIFT=	bi_ccd_drift.o bi_ccd_func.o bi_ccd_Si_funcs.o xray_bi_ccd.o \
		ray_util.o henke_utils.o multilayer.o complex_n_server.o \
		$(NUMLIB)hunt.o \
		$(NUMLIB)nrutil.o $(NUMLIB)ran1.o $(NUMLIB)expdev.o \
		$(NUMLIB)gasdev.o $(NUMLIB)poidev.o $(NUMLIB)gammln.o \
		$(NUMLIB)ran2.o $(NUMLIB)bnldev.o $(NUMLIB)bessk1.o \
		$(NUMLIB)bessi1.o 

bi_ccd_drift: 	$(BI_CCD_DRIFT) ray.h henke.h bi_ccd.h complex_n_server.h
		$(LINK.c) -o $@ $(BI_CCD)

generate_sens_luts.o:	lsst_fp.c lsst_fp.h camera_geometry.h \
			generate_sens_luts.c
		$(COMPILE.c) -o $@ generate_sens_luts.c

GEN_SENS_LUTS_ORIG=	lsst_fp.o generate_sens_luts.o bi_ccd_func.o \
			henke_utils.o \
			zernike.o $(NUMLIB)gasdev.o $(NUMLIB)ran1.o \
			$(NUMLIB)poidev.o $(NUMLIB)expdev.o $(NUMLIB)hunt.o \
			$(RAYUTIL)

GEN_SENS_LUTS=	lsst_fp.o generate_sens_luts.o zernike.o \
		bi_ccd_func.o bi_ccd_Si_funcs.o henke_utils.o \
		multilayer.o complex_n_server.o \
		$(NUMLIB)ran1.o \
		$(NUMLIB)ran2.o \
		$(NUMLIB)gasdev.o \
		$(NUMLIB)poidev.o \
		$(NUMLIB)expdev.o \
		$(NUMLIB)hunt.o \
		$(RAYUTIL)

generate_sens_luts:	$(GEN_SENS_LUTS)
		$(LINK.c) -o $@ $(GEN_SENS_LUTS) $(LDFLAGS)

field_superpose.o:	field_superpose.c field_superpose.h

lsst_fp.o:	lsst_fp.c lsst_fp.h bi_ccd_func.c bi_ccd.h sensor_pars.h \
		simulated_LSST_fp_chip_errors.h camera_geometry.h ray.h

LSST_CCSXY2INDICES=	lsst_fp.o bi_ccd_func.o bi_ccd_Si_funcs.o \
		henke_utils.o zernike.o multilayer.o complex_n_server.o \
		xray_bi_ccd.o \
		$(NUMLIB)bnldev.o $(NUMLIB)ran2.o \
		$(NUMLIB)bessk1.o \
		$(NUMLIB)bessi1.o \
		$(NUMLIB)hunt.o $(NUMLIB)factrl.o $(NUMLIB)ran1.o \
		$(NUMLIB)nrutil.o $(NUMLIB)gammln.o $(NUMLIB)gasdev.o \
		$(NUMLIB)poidev.o $(NUMLIB)expdev.o \
		ray_util.o lsst_ccsxy2indices.o 

lsst_ccsxy2indices:	$(LSST_CCSXY2INDICES)
		$(LINK.c) -o $@ $(LSST_CCSXY2INDICES)

LSST_FP.o:	LSST_FP.c lsst_fp.c zernike.c lsst_fp.h \
		ray.h bi_ccd_func.c bi_ccd_Si_funcs.c
		$(COMPILE.c) -o $@ LSST_FP.c

LSST_FP=        LSST_FP.o lsst_fp.o zernike.o multilayer.o complex_n_server.o \
		bi_ccd_func.o bi_ccd_Si_funcs.o henke_utils.o \
		$(NUMLIB)hunt.o $(NUMLIB)gasdev.o $(NUMLIB)poidev.o \
		$(NUMLIB)ran2.o $(NUMLIB)ran1.o $(NUMLIB)expdev.o $(RAYUTIL)

LSST_FP: 	$(LSST_FP) camera_geometry.h ray.h
		$(LINK.c) -o $@ $(LSST_FP)

# consider structural change in this software to remove commonality
# between this and bi_ccd.. in principle asphere has little to do with bi_ccd

ASPHERE=	asphere.o ray_util.o zernike.o multilayer.o complex_n_server.o \
		bi_ccd_Si_funcs.o henke_utils.o xray_bi_ccd.o f_2d_tree.o \
		$(NUMLIB)gasdev.o $(NUMLIB)bnldev.o \
		$(NUMLIB)poidev.o $(NUMLIB)bessk1.o $(NUMLIB)ran1.o $(NUMLIB)bessi1.o \
		$(NUMLIB)expdev.o $(NUMLIB)hunt.o $(NUMLIB)nrutil.o $(NUMLIB)ran2.o \
		$(NUMLIB)factrl.o $(NUMLIB)gammln.o

asphere: 	$(ASPHERE) ray.h zernike.h f_2d_tree.h
		$(LINK.c) -o $@ $(ASPHERE)

TEST_ZERNIKE=	test_zernike.o ray_util.o zernike.o $(NUMLIB)nrutil.o $(NUMLIB)ran2.o $(NUMLIB)factrl.o $(NUMLIB)gammln.o

test_zernike: 	$(TEST_ZERNIKE) ray.h zernike.h
		$(LINK.c) -o $@ $(TEST_ZERNIKE)

ASPHERE_T=	asphere_thermal.o ray_util.o zernike.o $(NUMLIB)nrutil.o $(NUMLIB)ran2.o $(NUMLIB)factrl.o $(NUMLIB)gammln.o

asphere_thermal: 	$(ASPHERE_T) ray.h zernike.h
		$(LINK.c) -o $@ $(ASPHERE_T)

BAFFLE=		baffle.o ray_util.o $(NUMLIB)nrutil.o $(NUMLIB)ran2.o

baffle: 	$(BAFFLE) ray.h
		$(LINK.c) -o $@ $(BAFFLE)

RAYLEIGH_MIE= 	rayleigh_mie.o ray_util.o \
		$(NUMREC)poidev.o $(NUMREC)ran1.o $(NUMREC)ran2.o \
		$(NUMREC)gammln.o $(NUMREC)nrutil.o

rayleigh_mie:	$(RAYLEIGH_MIE) ray.h
		$(LINK.c) -o $@ $(RAYLEIGH_MIE)

PLNCOL= 	plane_collapse.o ray_util.o 

plane_collapse:	$(PLNCOL) ray.h
		$(LINK.c) -o $@ $(PLNCOL)

TRANRAY= 	tran_ray.o ray_util.o 

tran_ray: 	$(TRANRAY) ray.h
		$(LINK.c) -o $@ $(TRANRAY)

SCALE_RAY_P= 	scale_ray_p.o $(RAYUTIL)

scale_ray_p: 	$(SCALE_RAY_P) ray.h
		$(LINK.c) -o $@ $(SCALE_RAY_P)

CHANGE_WAVE= 	change_wave.o $(RAYUTIL)

change_wave: 	$(CHANGE_WAVE) ray.h
		$(LINK.c) -o $@ $(CHANGE_WAVE)

SRCGAL= 	src_gal.o ray_util.o $(NUMLIB)nrutil.o \
		$(NUMLIB)ran1.o $(NUMLIB)hunt.o \
		$(NUMLIB)expdev.o

src_gal: 	$(SRCGAL) ray.h
		$(LINK.c) -o $@ $(SRCGAL)

SRCRAY= 	src_ray.o ray_util.o $(NUMLIB)nrutil.o $(NUMLIB)ran2.o \
		$(NUMLIB)ran1.o $(NUMLIB)poidev.o $(NUMLIB)gammln.o \
		$(NUMLIB)expdev.o

src_ray: 	$(SRCRAY) ray.h
		$(LINK.c) -o $@ $(SRCRAY)

INTERP=		interp.o $(NUMLIB)nrutil.o $(NUMLIB)hunt.o $(NUMLIB)polint.o

interp:		$(INTERP)
		$(LINK.c) -o $@ $(INTERP)

PRINTRAYS= 	print_rays.o ray_util.o 

print_rays: 	$(PRINTRAYS) ray.h
		$(LINK.c) -o $@ $(PRINTRAYS)

PRINTWAVE= 	print_wave.o ray_util.o 

print_wave: 	$(PRINTWAVE) ray.h
		$(LINK.c) -o $@ $(PRINTWAVE)

REVERSE_RAY= 	reverse_ray.o ray_util.o

reverse_ray: 	$(REVERSE_RAY) ray.h
		$(LINK.c) -o $@ $(REVERSE_RAY)

FINDFOCUS=	find_focus.o ray_util.o 

find_focus:	$(FINDFOCUS) ray.h
		$(LINK.c) -o $@ $(FINDFOCUS)

WFE=		wfe.o ray_util.o

wfe:		$(WFE)
		$(LINK.c) -o $@ $(WFE)

PSF_PARS=	psf_pars.o ray_util.o 

psf_pars:	$(PSF_PARS) ray.h
		$(LINK.c) -o $@ $(PSF_PARS)

DCD=		draw_catalog_dist.o $(NUMLIB)poidev.o $(NUMLIB)ran1.o \
		$(NUMLIB)gammln.o $(NUMLIB)nrutil.o ray_util.o

draw_catalog_dist:	$(DCD)
			$(LINK.c) -o $@ $(DCD)

nrutil.o:	$(NUMREC)nrutil.c
		$(COMPILE.c) -o numrec/lib/$@

setup_raytrace.bash:	generate_setup.perl
		./generate_setup.perl

TARGETS=diffuser image bi_ccd LSST_FP lsst_ccsxy2indices asphere baffle \
	plane_collapse tran_ray change_wave src_ray print_rays print_wave \
	generate_sens_luts reverse_ray find_focus psf_pars wfe lsst_fp_demux \
	setup_raytrace.bash 

all:	$(TARGETS)

.csh:
	cp $< $@
	chmod +x $@
	mv $@ $(BINDIR) 

.perl:
	cp $< $@
	chmod +x $@
	mv $@ $(BINDIR)

.sh:
	cp $< $@
	chmod +x $@
	mv $@ $(BINDIR)

%.o:	%.c rgs.h
	$(COMPILE.c) -o $@ $<

.c:     $<
	$(LINK.c) -o $(BINDIR)/$@ $< $(LDFLAGS)

$(NUMLIB)%.o:	$(NUMREC)%.c
		$(COMPILE.c) $(NRFLAGS) -o $@ $<

.for:	$<
	f77 $(FFLAGS) -o $(BINDIR)/$@ $< 

clean:
	rm -f *.o $(PROGRAMS)

