#define HENKE_STRLEN 1024
#define MAX_HENKE_ENTRIES 512

typedef struct henke {
  char  filename[HENKE_STRLEN];
  float energy[MAX_HENKE_ENTRIES];
  float f1[MAX_HENKE_ENTRIES];
  float f2[MAX_HENKE_ENTRIES];
  int   n_entries;
} henke_file;

typedef struct {
  float t_mgf;
  float t_al;
  float t_sio2;
  float t_si;
  float t_dep;
  float t_alox;
} ccd_str;

void interp_henke(henke_file *h,float energy,float *f1,float *f2);
void read_henke_file(henke_file *h);
float ccd_effic (float,ccd_str*); 

#define henke_dir "HENKE_DB_DIR"

