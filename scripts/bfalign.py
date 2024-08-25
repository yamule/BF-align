import re,os,sys;
import numpy as np;
import gzip;       
import torch


class PDBAtom:
    
    def __init__(self,line):
        line += "                                    ";
        self.head = line[0:6];
        self.serial_number = int(line[6:11]);
        self.atom_name = re.sub(" ","",line[12:16]);
        self.alt_loc = line[16];
        self.residue_name = re.sub("[\s]","",line[17:20]);
        self.chain_id = line[21];
        self.residue_pos = int(line[22:26]);
        self.insertion_code = line[26];
        self.x = float(line[30:38]);
        self.y = float(line[38:46]);
        self.z = float(line[46:54]);
        self.occupancy = line[54:60];
        self.bfactor = line[60:66];
        self.element = line[76:78];
        self.charge = line[79:80];
    
    #alt pos 以外を入れる
    def get_atom_label(self):
        return self.chain_id+"#"+self.residue_name+"#"+str(self.residue_pos)+"#"+self.insertion_code+"#"+self.atom_name;
    
    def get_residue_label(self):
        return self.chain_id+"#"+self.residue_name+"#"+str(self.residue_pos)+"#"+self.insertion_code;
    
    def distance(self,target):
        xx = self.x-target.x;
        yy = self.y-target.y;
        zz = self.z-target.z;
        rr = xx*xx+yy*yy+zz*zz;
        if rr == 0.0:
            return 0.0;
        return math.sqrt(rr);
        
    def make_line(self):
        xx = "{:>.3f}".format(self.x);
        yy = "{:>.3f}".format(self.y);
        zz = "{:>.3f}".format(self.z);
        
        if len(xx) > 8:
            raise Exception("string overflow".format(self.x));
        if len(yy) > 8:
            raise Exception("string overflow".format(self.y));
        if len(zz) > 8:
            raise Exception("string overflow".format(self.z));
        atomname = self.atom_name;
        if self.head == "ATOM  " and len(self.atom_name) < 4:
            atomname = " "+atomname;
        ret = "{head}{serial_number:>5} {atom_name:<4}{alt_loc}{residue_name:>3} {chain_id}{residue_pos:>4}{insertion_code}   {xx:>8}{yy:>8}{zz:>8}{occupancy:>6}{bfactor:>6}          {element:>2} {charge:2}".format(        
            head = self.head,
            serial_number = self.serial_number,
            atom_name = atomname,
            alt_loc = self.alt_loc,
            residue_name = self.residue_name,
            chain_id = self.chain_id,
            residue_pos = self.residue_pos,
            insertion_code = self.insertion_code,
            xx = xx,
            yy = yy,
            zz = zz,
            occupancy = self.occupancy,
            bfactor = self.bfactor,
            element = self.element,
            charge = self.charge);
        return ret;

def normalize(arr):
    dis = torch.sqrt(((arr)*(arr)).sum(axis=-1));
    return arr/dis[:,None];

def calc_norm(a):
    s = a[:,0,1]*a[:,1,2] - a[:,1,1]*a[:,0,2];
    t = -(a[:,0,0]*a[:,1,2] - a[:,1,0]*a[:,0,2]);
    u = a[:,0,0]*a[:,1,1] - a[:,1,0]*a[:,0,1];
    return torch.stack([s,t,u],axis=-1);

def dist(a,b):
    c = a-b;
    return torch.sqrt((c*c).sum(axis=-1));

def pos_to_frame(arr):
    assert len(arr.shape) == 3;
    n_ca = normalize(arr[:,0]-arr[:,1]);
    c_ca = normalize(arr[:,2]-arr[:,1]);
    norm = normalize(calc_norm(torch.stack([n_ca,c_ca],axis=1)));
    norm2 =  normalize(calc_norm(torch.stack([n_ca,norm],axis=1)));
    
    # print(dist(norm,n_ca));
    # print(dist(norm,norm2));
    # print(dist(n_ca,norm2));
    return torch.stack([n_ca,norm,norm2],axis=1),arr[:,1];

def load_atoms(infile,n_ca_c=False):
    if infile.endswith("pdb.gz"):
        fin = gzip.open(infile,"rt");
    else:
        fin = open(infile,"rt");
    alllines = fin.readlines();
    fin.close();
    residue_atoms = {};
    if n_ca_c:
        targets = set(["N","CA","C"]);
    else:
        targets = None;
    altloc = set(["A"," ",".","?"]);
    modelnum = "0";
    for li in range(len(alllines)):
        ll = re.sub("[\r\n]","",alllines[li])
        head = ll[0:6] ;
        if head == "MODEL ":
            mat = re.search(r"^MODEL[\s]+([^\s]+)",ll);
            modelnum = mat.group(1);
        if head == "ATOM  ":
            att = PDBAtom(ll);
            if att.alt_loc not in altloc:
                continue;
            if targets is None or att.atom_name in targets:
                lab = modelnum+"#"+att.get_residue_label();
                if lab not in residue_atoms:
                    residue_atoms[lab] = {};
                    residue_atoms[lab]["idx"] = len(residue_atoms)-1;
                if att.atom_name in residue_atoms[lab]:
                    sys.stderr.write(att.atom_name+" was already found in "+lab+". Please check format.\n"+ll+"\n");
                    raise Exception();
                residue_atoms[lab][att.atom_name] = att;
    lkey = list(sorted(residue_atoms.keys(),key=lambda x:residue_atoms[x]["idx"]));
    ret = [];
    for ll in list(lkey):
        pdic = residue_atoms[ll];
        if n_ca_c:
            if "N" in pdic and "CA" in pdic and "C" in pdic:
                ret.append(pdic);
        else:
            ret.append(pdic);
    return ret;

file1 = sys.argv[1];
file2 = sys.argv[2];
outfile = sys.argv[3];
ncac1 = load_atoms(file1,n_ca_c=True);
ncac2 = load_atoms(file2,n_ca_c=True);

len1 = len(ncac1);
len2 = len(ncac2);
pos1 = [];
for ff in list(ncac1):
    pos1.append([
        [ff["N"].x,ff["N"].y,ff["N"].z],
        [ff["CA"].x,ff["CA"].y,ff["CA"].z],
        [ff["C"].x,ff["C"].y,ff["C"].z]
    ]);
pos1 = torch.tensor(pos1).to("cuda");
rot1, trans1 = pos_to_frame(pos1);

revframe1 = rot1.permute(0,2,1);

test = False;
if test:
    pos1_2d = pos1[:,None]-pos1[None,:,1:2];
    pos1_2d = torch.einsum("stqp,tpx->stqx",pos1_2d,revframe1);
    for pp in range(pos1_2d.shape[0]):
        assert (torch.abs(pos1_2d[pp,pp,:,1]) < 0.0001).all();
    exit(0);

capos1_2d = pos1[:,None,1]-pos1[None,:,1];
capos1_2d = torch.einsum("stp,tpx->stx",capos1_2d,revframe1);


pos2 = [];
for ff in list(ncac2):
    pos2.append([
        [ff["N"].x,ff["N"].y,ff["N"].z],
        [ff["CA"].x,ff["CA"].y,ff["CA"].z],
        [ff["C"].x,ff["C"].y,ff["C"].z]
    ]);
pos2 = torch.tensor(pos2).to("cuda");
rot2,trans2 = pos_to_frame(pos2);
capos1_2d = torch.einsum("stp,upx->sutx",capos1_2d,rot2);
capos1_2d += trans2[None,:,None,:];

@torch.jit.script
def calc_rmsdsum(a,b):
    rmsd = a-b;
    rmsd = torch.sqrt(torch.sum((rmsd*rmsd),dim=-1));
    briefcheck = torch.sum((torch.min(rmsd,dim=-1)[0] < 3.0).to(torch.int32),dim=-1);
    return briefcheck;
"""
/* 
 * Parameters are from
 * 
 * References to cite:
 * Y Zhang, J Skolnick. Proteins, 57:702-10 (2004)
 *
 * DISCLAIMER:
 *  Permission to use, copy, modify, and distribute the Software for any
 *  purpose, with or without fee, is hereby granted, provided that the
 *  notices on the head, the reference information, and this copyright
 *  notice appear in all copies or substantial portions of the Software.
 *  It is provided "as is" without express or implied warranty.
 *
*/
"""
@torch.jit.script
def calc_tmscore(a,b,lnorm):
    D0_MIN = torch.tensor(0.5);

    d0=(1.24*pow((lnorm*1.0-15), 1.0/3)-1.8);
    if (lnorm<=21):
        d0=torch.tensor(0.5);
    
    if (d0<D0_MIN):
        d0=D0_MIN;   

    d0_search=d0;
    if (d0_search>8):
        d0_search=torch.tensor(8);
    
    if (d0_search<4.5):
        d0_search=torch.tensor(4.5);
    
    d02 = d0*d0;
    d0_search2 = d0_search*d0_search;

    rmsd = a-b;
    rmsd = torch.sum((rmsd*rmsd),dim=-1);
    if len(a.shape) > 2:
        mask = torch.zeros_like(rmsd).scatter(-1, (-1.0*rmsd).argmax(-1,True), value=1);
        mask = (rmsd <= d0_search2).to(torch.int32)*mask;
    else:
        mask = (rmsd <= d0_search2).to(torch.int32);
    tmscores = (((1.0/(1.0+rmsd/d02))*mask).sum(dim=-1)).sum(dim=-1)/lnorm;
    
    return tmscores;


capos1_2d = capos1_2d.to("cuda")
pos2 = pos2.to("cuda")
print(capos1_2d.shape)
max_score = 0.0;
max_index = (0,0);
for ii in range(capos1_2d.shape[0]):
    briefcheck = calc_tmscore(capos1_2d[ii],pos2[:,None,1:2],capos1_2d.shape[0])
    maxx = float(briefcheck.max().detach().cpu());
    if maxx > max_score:
        max_score = maxx;
        max_index = (ii,briefcheck.argmax());
    #break;

print("max_score:",max_score,"max_index:",max_index)

i1 = max_index[0]
i2 = max_index[1]
res_rot1 = rot1[i1];
res_trans1 = trans1[i1];
res_trans1 = torch.squeeze(torch.einsum("ab,bc->ac",res_rot1,res_trans1[:,None]));
print(res_trans1.shape)
print(res_rot1.shape)
print(pos1.shape)
pos_1b = torch.einsum("ab,cdb->cda",res_rot1,pos1) - res_trans1;
# print(pos_1b[i1]);
res_rot2 = rot2[i2];

rotp = torch.einsum("ab,bc->ac",res_rot2.permute(1,0),res_rot1);
pos_1b = torch.einsum("ab,cdb->cda",rotp,pos1);
transp = pos2[i2,1] - pos_1b[i1,1];

allresidues = load_atoms(file1);
allatoms = [];
allpositions = [];
for rr in list(allresidues):
    for aa_ in list(rr.keys()):
        if aa_ == "idx":
            continue;
        aa = rr[aa_];
        allatoms.append(aa);
        allpositions.append(
            [aa.x,aa.y,aa.z]
        );
allpositions = torch.tensor(allpositions).to("cuda");
res = torch.einsum(
    "ab,cb->ca",rotp,allpositions
);
res += transp;
with open(outfile,"wt") as fout:
    for ii in range(len(allatoms)):
        atom = allatoms[ii];
        atom.x = res[ii,0];
        atom.y = res[ii,1];
        atom.z = res[ii,2];
        fout.write(atom.make_line()+"\n");

# 二つの点群の全点 vs 全点で距離を計算し、apos から最も近い bpos の点のインデクスが入ったリストを返す。
# 同じ点にマップされた場合は、距離の近い方のみ返す。マップできなかった場合 None が入っている。
# maxsqdist は二乗された距離
def get_mapping(apos,bpos,maxsqdist):
    dis = apos[:,None]-bpos[None,:];
    dis = (dis*dis).sum(dim=-1);
    idx = (-1.0*dis).argmax(dim=-1).detach().cpu();
    mapper_rev = {};
    mapper = [None for ii in range(dis.shape[0])];
    updated = [ii for ii in range(dis.shape[0])];
    while len(updated) > 0:
        updated_new = [];
        while len(updated) > 0:
            ii = updated.pop();
            if len(idx) > 0:
                amax = int(idx[ii]);
            else:
                mindis = 999;
                amax = None;
                
                # dist2(CA_A1, CA_B1) > dist2(CA_A1, CA_B2)
                # dist2(CA_A2, CA_B2) > dist2(CA_A1, CA_B2)
                # dist2(CA_A1, CA_B1) < maxsqdist
                # dist2(CA_A1, CA_B2) < maxsqdist
                # dist2(CA_A2, CA_B1) > maxsqdist
                # dist2(CA_A2, CA_B2) < maxsqdist
                # のようなケースの時、最適なマッピングは CA_A1->CA_B1, CA_A2->CA_B2
                # だが、そのようなケースには対応してない。CA_A1->CA_B2, CA_A2->None になる。

                for jj in range(dis.shape[1]):
                    ddis =  dis[ii,jj]
                    if jj in mapper_rev and dis[mapper_rev[jj],jj] < ddis:
                        continue;
                    if mindis > ddis:
                        mindis = ddis;
                        amax = jj;
            if amax is None:
                mapper[ii] = None;
                continue;
            if dis[ii,amax] > maxsqdist:
                mapper[ii] = None;
                continue;
            if amax in mapper_rev:
                if dis[ii,amax] < dis[mapper_rev[amax],amax]:
                    mapper[mapper_rev[amax]] = None;
                    updated_new.append(mapper_rev[amax]);
                else:
                    amax = None;
                    updated_new.append(ii);
            if amax is not None:
                mapper_rev[amax] = ii;
            mapper[ii]  = amax;
        updated = updated_new;
        idx = [];

    assert len(mapper) == apos.shape[0]
    

    return mapper;

def get_mapped_arrays(p1,p2,mapper):
    apos = [];
    bpos = [];
    for ii in range(len(mapper)):
        jj = mapper[ii];
        if jj is None:
            continue;
        apos.append(
            [p1[ii,0],p1[ii,1],p1[ii,2]]
        );
        bpos.append(
            [p2[jj,0],p2[jj,1],p2[jj,2]]
        );
    return apos, bpos;

if True:
    from Bio.SVDSuperimposer import SVDSuperimposer ;
    import copy;
    maxsqdist = 9.0*9.0;

    allcas_ = [];
    for ii in range(len(allatoms)):
        atom = allatoms[ii];
        if atom.atom_name == "CA":
            allcas_.append(copy.deepcopy(atom));

    for _ in range(5):
        allcas = [[a.x,a.y,a.z] for a in list(allcas_)];
        
        allcas = torch.tensor(allcas,device="cuda");
        mapper = get_mapping(allcas,pos2[:,1],maxsqdist);
        print(mapper)

        apos,bpos = get_mapped_arrays(allcas.detach().cpu().numpy(),pos2[:,1].detach().cpu().numpy(),mapper);
        #print(np.array(apos).shape,np.array(bpos).shape);
        #print(apos,bpos)
        sup = SVDSuperimposer();
        sup.set(np.array(bpos),np.array(apos));
        sup.run();
        rot,trans = sup.get_rotran();
        rot = torch.tensor(rot,device="cuda");
        print(rot);
        trans = torch.tensor(trans,device="cuda");
        allcas = torch.einsum("ab,bc->ac",allcas,rot)+trans;
        mapper = get_mapping(allcas,pos2[:,1],maxsqdist)
        print(mapper)
        apos,bpos = get_mapped_arrays(allcas,pos2[:,1],mapper);
        apos = torch.tensor(apos).to("cuda")
        bpos = torch.tensor(bpos).to("cuda")
        tmscore1 = calc_tmscore(apos,bpos,len1);
        tmscore2 = calc_tmscore(apos,bpos,len2);
        print(tmscore1)
        print(tmscore2)
        for ii in range(len(allcas)):
            allcas_[ii].x = allcas[ii,0];
            allcas_[ii].y = allcas[ii,1];
            allcas_[ii].z = allcas[ii,2];
    
    allpositions = [];
    for aa in list(allatoms):
        allpositions.append(
            [aa.x,aa.y,aa.z]
        );
    allpositions = torch.tensor(allpositions).to("cuda");

    res = torch.einsum(
        "cb,ba->ca",allpositions,rot
    )+trans;

    with open(outfile+".svd.pdb","wt") as fout:
        for ii in range(len(allatoms)):
            atom = allatoms[ii];
            atom.x = res[ii,0];
            atom.y = res[ii,1];
            atom.z = res[ii,2];

            fout.write(atom.make_line()+"\n");