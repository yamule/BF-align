import re,os,sys;
import numpy as np;
import gzip;       
import torch;
import argparse;

aa_3_1_ = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E"
,"GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P"
,"SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"};

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
    
    # Return labels without alt_loc.
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

def dist2(apos,bpos):
    dis = apos[:,None]-bpos[None,:];
    dis = (dis*dis).sum(dim=-1);
    return dis;

# Construct local axis from 3 points.
# I can't remember where I learned this algorithm. It may be more than 10 years ago.
# I think in some tutorials of Three.js or OpenGL.
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

def load_atoms(infile,targets=None):
    if infile.endswith("pdb.gz"):
        fin = gzip.open(infile,"rt");
    else:
        fin = open(infile,"rt");
    alllines = fin.readlines();
    fin.close();
    residue_atoms = {};
    
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
        if targets is not None:
            assert "idx" in pdic;
            if len(pdic.keys()) == len(targets)+1:
                ret.append(pdic);
        else:
            ret.append(pdic);
    return ret;

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


# Calculate distances between all points in apos and all points in bpos, and return a list of indices of the closest points in bpos for each point in apos.
# If multiple points are mapped to the same point, only the one with the shortest distance is returned. If a point couldn't be mapped, None is inserted.
# 'maxsqdist' is the squared distance
def get_mapping(apos,bpos,maxsqdist):
    
    dis = dist2(apos,bpos);
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
                
                # In cases like:
                # dist2(CA_A1, CA_B1) > dist2(CA_A1, CA_B2)
                # dist2(CA_A2, CA_B2) > dist2(CA_A1, CA_B2)
                # dist2(CA_A1, CA_B1) < maxsqdist
                # dist2(CA_A1, CA_B2) < maxsqdist
                # dist2(CA_A2, CA_B1) > maxsqdist
                # dist2(CA_A2, CA_B2) < maxsqdist
                # The optimal mapping would be CA_A1->CA_B1, CA_A2->CA_B2
                # However, this case is not handled. It will result in CA_A1->CA_B2, CA_A2->None.

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

# 'mapper' is an array of p1[index]->p2index which indicates the closest atom paires. If a point in p1 is not mapped to p2, it's 'None'.
# Returns two arrays with same length, where mapped points are at the same index.
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

# pos1 and pos2 are torch.tensor which have shapes
# (N, 3, 3)
# The first dimension is length of a protein.
# The second dimension specifies three atoms which construct triangles to align, which will typicall be backbone atoms; N, CA, C.
# The last dimensions have xyz coordinates.
# If realign == True, it performs structural alignment using points aligned with Biopython's SVDSuperimposer.
def bfalign(pos1,pos2,realign=True):
    ddev = pos1.device
    rot1, trans1 = pos_to_frame(pos1);

    revframe1 = rot1.permute(0,2,1);

    capos1_2d = pos1[:,None,1]-pos1[None,:,1];
    capos1_2d = torch.einsum("stp,tpx->stx",capos1_2d,revframe1);

    rot2,trans2 = pos_to_frame(pos2);
    capos1_2d = torch.einsum("stp,upx->sutx",capos1_2d,rot2);
    capos1_2d += trans2[None,:,None,:];


    capos1_2d = capos1_2d
    pos2 = pos2

    max_score1 = 0.0;
    max_score2 = 0.0;
    max_index = (0,0);
    for ii in range(capos1_2d.shape[0]):
        briefcheck = calc_tmscore(capos1_2d[ii],pos2[:,None,1:2],capos1_2d.shape[0])
        briefcheck2 = calc_tmscore(capos1_2d[ii],pos2[:,None,1:2],pos2.shape[0])
        maxx = float(briefcheck.max().detach().cpu());
        if maxx > max_score1:
            max_score1 = maxx;
            max_score2 = float(briefcheck2.max().detach().cpu());
            max_index = (ii,briefcheck.argmax());
        
    # print("max_score:",max_score1,"max_index:",max_index)

    i1 = max_index[0]
    i2 = max_index[1]
    res_rot1 = rot1[i1];
    res_trans1 = trans1[i1];
    res_trans1 = torch.squeeze(torch.einsum("ab,bc->ac",res_rot1,res_trans1[:,None]));
    res_rot2 = rot2[i2];

    rotp = torch.einsum("ab,bc->ac",res_rot2.permute(1,0),res_rot1);
    pos_1b = torch.einsum("ab,cdb->cda",rotp,pos1);
    transp = pos2[i2,1] - pos_1b[i1,1];
    pos_1b += transp;
    if not realign:
        return {"tmscore1":max_score1,"tmscore2":max_score2,"rot":rotp.permute(1,0),"trans":transp};

    from Bio.SVDSuperimposer import SVDSuperimposer ;
    import copy;
    maxsqdist = 8.0*8.0;

    allcas_ = copy.deepcopy(pos1[:,1]);
    allcas = pos_1b[:,1];
    maxmap = None;
    maxscore1 = 0.0;
    maxscore2 = 0.0;
    maxmat = None;

    mapper = get_mapping(allcas,pos2[:,1],maxsqdist);
    sup = SVDSuperimposer();
    for _ in range(5):
        allcas = copy.deepcopy(allcas_);
        apos,bpos = get_mapped_arrays(allcas.detach().cpu().numpy(),pos2[:,1].detach().cpu().numpy(),mapper);
        sup.set(np.array(bpos),np.array(apos));
        sup.run();
        rot,trans = sup.get_rotran();
        rot = torch.tensor(rot,device=ddev);
        trans = torch.tensor(trans,device=ddev);
        allcas = torch.einsum("ab,bc->ac",allcas,rot)+trans;
        mapper = get_mapping(allcas,pos2[:,1],maxsqdist)
        apos,bpos = get_mapped_arrays(allcas,pos2[:,1],mapper);
        apos = torch.tensor(apos,device=ddev);
        bpos = torch.tensor(bpos,device=ddev);
        tmscore1 = calc_tmscore(apos,bpos,len1);
        tmscore2 = calc_tmscore(apos,bpos,len2);
        
        # Update if the alignment was improved.
        tmscore1 = float(tmscore1.detach().cpu())
        # print(tmscore1," vs ",maxscore1)
        if tmscore1 > maxscore1:
            tmscore2 = float(tmscore2.detach().cpu())
            maxmat = (rot,trans);
            maxscore1 = tmscore1;
            maxscore2 = tmscore2;
            maxmap = mapper;
        else:
            break;
    return {"tmscore1":maxscore1,"tmscore2":maxscore2,"rot":maxmat[0],"trans":maxmat[1]};


def check_bool(v):
    v = v.lower();
    if v == "true" or v == "1":
        return True;
    if v == "false" or v == "0":
        return False;
    raise Exception("true or false or 1 or 0 are expected.");

if __name__=="__main__":
    parser = argparse.ArgumentParser(prog="BF-align",description="Fully sequence independent structure alignment.");
    parser.add_argument("--file1",help='Query protein structure in PDB format to be aligned to file2 structure.',required=True) ;
    parser.add_argument("--file2",help='Template protein structure in PDB format to align against.',required=True) ;
    parser.add_argument("--outfile",help='Output file path of alignment result of file1 structure.',required=False,default=None);
    parser.add_argument("--use_ca",help='Use 3 CA atoms for alignment.',required=False,default=False,type=check_bool);
    parser.add_argument("--realign",help='Perform re-alignment with Biopython\'s SVDSuperimposer. ',required=False,default=False,type=check_bool);
    parser.add_argument("--device",help='Computation device: \'cpu\' or \'cuda\'.',required=False,default="cpu");

    args = parser.parse_args();

    file1 = args.file1;
    file2 = args.file2;
    realign = args.realign;
    outfile = args.outfile;
    ddev = args.device;
    use_ca = args.use_ca;

    seq1 = [];
    seq2 = [];
    pos1 = [];
    pos2 = [];

    if not use_ca:
        ncac1 = load_atoms(file1,targets=set(["N","CA","C"]));
        ncac2 = load_atoms(file2,targets=set(["N","CA","C"]));
        for aa in ncac1:
            seq1.append(aa_3_1_.get(aa["CA"].residue_name,"X"));
        for aa in ncac2:
            seq2.append(aa_3_1_.get(aa["CA"].residue_name,"X"));

        for ff in list(ncac1):
            pos1.append([
                [ff["N"].x,ff["N"].y,ff["N"].z],
                [ff["CA"].x,ff["CA"].y,ff["CA"].z],
                [ff["C"].x,ff["C"].y,ff["C"].z]
            ]);

        for ff in list(ncac2):
            pos2.append([
                [ff["N"].x,ff["N"].y,ff["N"].z],
                [ff["CA"].x,ff["CA"].y,ff["CA"].z],
                [ff["C"].x,ff["C"].y,ff["C"].z]
            ]);
    else:
        ca1 = load_atoms(file1,targets=set(["CA"]));
        ca2 = load_atoms(file2,targets=set(["CA"]));
        for aa in list(ca1):
            seq1.append(aa_3_1_.get(aa["CA"].residue_name,"X"));
        for aa in list(ca2):
            seq2.append(aa_3_1_.get(aa["CA"].residue_name,"X"));

        # Create a triangle using 3 consecutive (in the array, not in the peptide) CA atoms.
        # The first and last triangles are dummy triangles not expected to align.
        def ca_triangle(cas):
            ret = [];
            ret.append([
                [cas[2]["CA"].x,cas[2]["CA"].y,cas[2]["CA"].z],
                [cas[0]["CA"].x,cas[0]["CA"].y,cas[0]["CA"].z],
                [cas[1]["CA"].x,cas[1]["CA"].y,cas[1]["CA"].z]
            ]);
            for ii in range(1,len(cas)-1):
                ret.append([
                    [cas[ii-1]["CA"].x,cas[ii-1]["CA"].y,cas[ii-1]["CA"].z],
                    [cas[ii]["CA"].x,cas[ii]["CA"].y,cas[ii]["CA"].z],
                    [cas[ii+1]["CA"].x,cas[ii+1]["CA"].y,cas[ii+1]["CA"].z]
                ]);

            ret.append([
                [cas[-3]["CA"].x,cas[-3]["CA"].y,cas[-3]["CA"].z],
                [cas[-1]["CA"].x,cas[-1]["CA"].y,cas[-1]["CA"].z],
                [cas[-2]["CA"].x,cas[-2]["CA"].y,cas[-2]["CA"].z]
            ]);
            return ret;
        pos1 = ca_triangle(ca1);
        pos2 = ca_triangle(ca2);

    len1 = len(pos1);
    len2 = len(pos2);

    pos1 = torch.tensor(pos1).to(ddev);
    pos2 = torch.tensor(pos2).to(ddev);

    align_result = bfalign(pos1,pos2,realign=realign);
    rotp = align_result["rot"];
    transp = align_result["trans"];

    allresidues = load_atoms(file1);
    allatoms = [];
    allpositions = [];
    for rr in list(allresidues):
        for aa_ in list(rr.keys()):
            if aa_ == "idx":
                continue;
            aa = rr[aa_];
            allatoms.append(aa);
            allpositions.append([aa.x,aa.y,aa.z]);
    allpositions = torch.tensor(allpositions).to(ddev);
    res = torch.einsum(
        "ab,bc->ac",allpositions,rotp
    );
    res += transp;
    allcas = [];
    for eii,aa in enumerate(allatoms):
        if aa.atom_name == "CA":
            allcas.append(res[eii]);

    maxmap = get_mapping(torch.stack(allcas,dim=0),pos2[:,1],8.0*8.0);
    aseq = [];
    bseq = [];
    ii = 0;
    while ii < len(maxmap):
        if maxmap[ii] is not None:
            bstart = maxmap[ii];
            bend = maxmap[ii];
            aend = ii;
            for jj in range(ii+1,len(maxmap)):
                if maxmap[jj] is not None:
                    if maxmap[jj] > bend:
                        bend = maxmap[jj];
                        aend = jj;
                    else:
                        break;
            aseq_ = ["{:>5} ".format(str(ii+1))];
            bseq_ = ["{:>5} ".format(str(bstart+1))];
            bi = bstart -1;
            for jj in range(ii,aend+1):
                if maxmap[jj] is None:
                    aseq_.append(seq1[jj]);
                    bseq_.append("-");
                else:
                    while maxmap[jj] > bi+1:
                        aseq_.append("-");
                        bi += 1;
                        bseq_.append(seq2[bi]);
                    aseq_.append(seq1[jj]);
                    bi += 1;
                    bseq_.append(seq2[bi])
            aseq_.append(" "+str(aend+1));
            bseq_.append(" "+str(bend+1));
            aseq.append("".join(aseq_));
            bseq.append("".join(bseq_));
            ii = aend+1;
        else:
            ii += 1;
    if outfile is not None:
        with open(outfile,"wt") as fout:
            for ii in range(len(allatoms)):
                atom = allatoms[ii];
                atom.x = res[ii,0];
                atom.y = res[ii,1];
                atom.z = res[ii,2];
                fout.write(atom.make_line()+"\n");
    print("file1:",args.file1);
    print("file2:",args.file2);
    print("TM-score normalized by file1 structure:",align_result["tmscore1"]);
    print("TM-score normalized by file2 structure:",align_result["tmscore2"]);
    for aa,bb in zip(aseq,bseq):
        print("file1:",aa);
        print("file2:",bb);
        print();