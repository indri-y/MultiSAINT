# Parser.py — GLN FPGA (structural HT labeling)
import os, re, csv, argparse, collections
import numpy as np
import networkx as nx
import re
from collections import Counter
from scipy.sparse import csr_matrix, lil_matrix, save_npz

# ========= CONFIG =========
# Drop dari graph (biar gak jadi super-node)
SKIP_TYPES = {'VCC', 'GND', 'BUFG', 'IBUF', 'OBUF'}
# Net konstanta Vivado
CONST_NETS = {'\\<const0>', '\\<const1>'}
# Soft-hints nama (boleh dimatikan kalau mau pure struktural)
USE_ENA_HINT = False                 # set True kalau "ena" memang konsisten sebagai penanda HT
HT_NAME_RX   = re.compile(r'^\b$')   # mode strict (murni struktur). Artinya model belajar dari pola GLN‑nya 
# HT_NAME_RX   = re.compile(r'trojan|trigger|payload|t\d{2,4}', re.I)  # “menghafal” nama sinyal.
ENA_RX       = re.compile(r'(?<![a-z0-9])ena(?![a-z0-9])', re.I)

# Primitif yang dipakai heuristik
PRIMS_SEQ  = {'FDRE','FDSE','FDCE','FDPE'}
PRIMS_CARRY= {'CARRY4'}
PRIMS_MUX  = {'MUXF7','MUXF8'}
PRIMS_LUT  = {'LUT1','LUT2','LUT3','LUT4','LUT5','LUT6'}
# Sink (jalur sensitif) per family — regex disatukan
SINK_PATTERNS = re.compile(
    r'(cipher|state|^out(\[|\b)|sbox|round|'           # AES
    r'xmit_doneH|rec_readyH|(^tx_)|(^rx_)|'            # RS232
    r'(^ir_)|instr|(^pc_)|prog_addr|'                  # PIC16F84
    r'ciphertext|(^exp_)|exponent|'                    # RSA
    r'(^addr_)|address|(^instr_)|'                     # b19
    r'prio|arb|grant|slave_sel)', re.I
)
# ==========================

def flatten(l): return [item for sublist in l for item in sublist]

def getverilogs(path='./'):
    TRAIN, VAL, TEST = path+'train', path+'val', path+'test'
    trainids, valids, testids = os.listdir(TRAIN), os.listdir(VAL), os.listdir(TEST)
    # hanya ambil .v (case-insensitive)
    trainverilogs = [os.path.join(TRAIN, v) for v in trainids if v.lower().endswith('.v')]
    valverilogs   = [os.path.join(VAL,   v) for v in valids  if v.lower().endswith('.v')]
    testverilogs  = [os.path.join(TEST,  v) for v in testids if v.lower().endswith('.v')]
    return [trainverilogs, valverilogs, testverilogs]

def read_verilog_blocks(verilog_file):
    """Yield setiap instansiasi primitive sebagai satu string (multi-line safe)."""
    with open(verilog_file, 'r', encoding='utf-8', errors='ignore') as f:
        buf, inside = '', False
        for line in f:
            s = line.strip()
            if not inside:
                if any(s.lstrip().startswith(g) for g in
                       ("LUT","FDRE","FDSE","FDCE","FDPE","CARRY","MUXF","BUFG","OBUF","IBUF","GND","VCC","RAMB")):
                    buf, inside = line, True
                else:
                    continue
            else:
                buf += line
            if inside and ');' in s:
                yield buf.replace('\n',' ')
                buf, inside = '', False

def readlogs(verilog_list, verbose=False):
    ret = []
    for filename in verilog_list:
        temp = []
        for block in read_verilog_blocks(filename):
            temp.append(block)
            if verbose:
                first_tok = block.strip().split()[0] if block.strip() else ""
                if first_tok not in SKIP_TYPES:
                    print("BLOCK:", block[:100])
        ret.append(temp)
    return ret

def getmoduleinfo(path='./'):
    """
    Ambil header modul dari library.
    - Prioritas: verilog_fpga_lib.v (kalau ada), fallback ke verilog_lib.v
    - Tahan stub 1-baris maupun multi-baris (endmodule bisa di tengah baris).
    - Hasil: list of strings, masing2 satu blok "module ... endmodule" (tanpa newline).
    """

    lib_fpga = os.path.join(path, 'verilog_fpga_lib.v')
    lib_asic = os.path.join(path, 'verilog_lib.v')
    lib_file = lib_fpga if os.path.exists(lib_fpga) else lib_asic

    with open(lib_file, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    # Hapus komentar baris & atribut Vivado agar regex lebih simpel
    text = re.sub(r'//.*', '', text)
    text = re.sub(r'\(\*.*?\*\)', '', text, flags=re.S)

    # Tangkap semua blok module ... endmodule (termasuk yang 1-baris)
    blocks = re.findall(r'\bmodule\b[\s\S]*?\bendmodule\b', text, flags=re.I)

    # Rapikan: buang newline agar downstream parser konsisten
    blocks = [re.sub(r'\s+', ' ', b).strip() for b in blocks]
    return blocks


def getinfo(line):
    """Parse header 'module NAME(port0,port1,...)' dari library (port0=output, sisanya=input)."""
    line = line.strip().replace('\n','')
    m = re.match(r'^module\s+([A-Za-z0-9_]+)\s*\(\s*([^\)]*)\)', line)
    if not m: return None, None, None
    name, ports = m.group(1), [p.strip() for p in m.group(2).split(',') if p.strip()]
    if not ports: return name, None, None
    out = [ports[0]]; inp = ports[1:] if len(ports)>1 else None
    return name, inp, out

def parse_modules(module_list):
    modules = {}
    for i in module_list:
        name, inp, out = getinfo(i)
        modules[name] = (inp, out)
    return modules

def getlineinfo(line):
    """Parse satu instansiasi primitive (Vivado-friendly)."""
    line = line.replace("BLOCK:", "").strip()
    line = re.sub(r"\(\*.*?\*\)", "", line)                     # attributes
    line = re.sub(r",\s*\.INIT\([^)]*\)", "", line)             # INIT
    line = re.sub(r",\s*\.IS_[A-Z_]+\([^)]*\)", "", line)       # IS_*
    line = re.sub(r"//.*", "", line).strip()                    # comments
    line = line.strip().rstrip(";").replace("\n"," ").replace("\r"," ").strip()
    if not line: return None
    m = re.match(
        r"^([A-Za-z0-9_]+)"                # gate type
        r"(?:\s*#\s*\(.*?\))?"             # optional params
        r"\s+([\\A-Za-z0-9_\[\]\-\.]+)"    # instance name
        r"\s*\((.*)\)$",                   # ports (...)
        line
    )
    if not m: return None
    gatetype, instancename, ports_str = m.group(1), m.group(2), m.group(3)

    portnames, connectionnames, ports = [], [], []
    depth, buf = 0, ""
    for ch in ports_str:
        if ch == "(": depth += 1
        elif ch == ")": depth -= 1
        if ch == "," and depth == 0:
            ports.append(buf); buf = ""
        else: buf += ch
    if buf: ports.append(buf)

    for p in [p.strip() for p in ports]:
        zz = re.match(r"\.([A-Za-z0-9_]+)\s*\(\s*(\\[^ \t\)]+|[A-Za-z0-9_\[\]']+)\s*\)", p)
        if zz:
            portnames.append(zz.group(1)); connectionnames.append(zz.group(2))
    return (gatetype, gatetype, instancename, 0, portnames, connectionnames)

# -------- PRIMARY IO dari top-level tiap file --------
def get_top_module_ios(verilog_file):
    prim_inputs, prim_outputs, inside = set(), set(), False
    with open(verilog_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            l = line.strip().rstrip(';')
            if l.startswith("module ") and not inside: inside = True; continue
            if not inside: continue
            if l.startswith("input "):
                decl = l[len("input "):].strip()
                for p in [pp.strip() for pp in decl.split(',')]:
                    vb = re.match(r'\[(\d+):(\d+)\]\s*([A-Za-z0-9_]+)', p)
                    if vb:
                        msb, lsb, name = int(vb.group(1)), int(vb.group(2)), vb.group(3)
                        for i in range(lsb, msb+1): prim_inputs.add(f"{name}[{i}]")
                    else: prim_inputs.add(p)
            elif l.startswith("output "):
                decl = l[len("output "):].strip()
                for p in [pp.strip() for pp in decl.split(',')]:
                    vb = re.match(r'\[(\d+):(\d+)\]\s*([A-Za-z0-9_]+)', p)
                    if vb:
                        msb, lsb, name = int(vb.group(1)), int(vb.group(2)), vb.group(3)
                        for i in range(lsb, msb+1): prim_outputs.add(f"{name}[{i}]")
                    else: prim_outputs.add(p)
            elif l.startswith("endmodule"): break
    return prim_inputs, prim_outputs

def getprimlist_from_files(verilog_paths):
    primin, primout = [], []
    for split in verilog_paths:
        for fp in split:
            i, o = get_top_module_ios(fp)
            primin.append(i); primout.append(o)
    return primin, primout
# -----------------------------------------------------

def fixassigns(linelist):
    ret = []
    for split in linelist:
        subret = []
        for lines in split:
            subsubret, blacklist = [], {}
            for l in lines:
                if 'assign' in l:
                    temp = l.split(' ',1)[1]; temp = temp.split(' ',1)[1]; temp = temp.split(' ',1)[1]
                    leftside, rightside = temp.split(' = ')
                    blacklist[rightside] = leftside
                else:
                    for k in list(blacklist.keys()):
                        if k in l: l = l.replace(k, blacklist[k])
                    subsubret.append(l)
            subret.append(subsubret)
        ret.append(subret)
    return ret

def parse_lines(lines, veriloglistlist, modules):
    gates=set(); nodeslistlist=[]; infolistlist=[]; indexlistlist=[]
    trainindices, valindices, testindices = [], [], []
    i = 0
    for split,(nets,filelist) in enumerate(zip(lines, veriloglistlist)):
        nodeslist=[]; infolist=[]; indexlist=[]
        for net, filepath in zip(nets, filelist):
            nodes={}; info=[]
            for l in net:
                if l=='' or '//' in l or l=='\n': continue
                parsed = getlineinfo(l)
                if parsed is None: continue
                gt,gtt,inst,_,pnames,cnames = parsed
                if gt not in modules: continue
                if gt in SKIP_TYPES:  continue
                # label sementara 0 (akan diganti oleh heuristik struktural)
                x = (gt,gtt,inst,0,pnames,cnames)
                nodes[inst]=i; gates.add(gt); info.append(x)
                if   split==0: trainindices.append(i)
                elif split==1: valindices.append(i)
                else:          testindices.append(i)
                i+=1
            infolist.append(info)
            indexlist.append({v:k for k,v in nodes.items()})
            nodeslist.append(nodes)
        nodeslistlist.append(nodeslist)
        infolistlist.append(infolist)
        indexlistlist.append(indexlist)
    return gates, flatten(nodeslistlist), flatten(infolistlist), flatten(indexlistlist), (trainindices,valindices,testindices)

def generate_lookup(infolist, modules):
    lookuplist=[]
    for info in infolist:
        lookup={}
        for i,x in enumerate(info):
            gt, pnames, cnames = x[0], x[4], x[5]
            inp,_ = modules.get(gt,(None,None))
            if inp is None: continue
            for q,pn in enumerate(pnames):
                if pn in inp:
                    conn = cnames[q]
                    if conn in CONST_NETS: continue
                    lookup.setdefault(conn, []).append(i)
        lookuplist.append(lookup)
    return lookuplist

def connect(shape, infolist, lookuplist, modules, train_indices):
    adj = lil_matrix((shape,shape), dtype=bool)
    adj_tr = lil_matrix((shape,shape), dtype=bool)
    class_map={}, {}; membership={}; num_neighbs=np.zeros(shape, dtype=int)  # class_map will be filled later
    class_map = {}
    i=0; netlist=0
    for info,lookup in zip(infolist, lookuplist):
        offset=i
        for x in info:
            gt, inst, pnames, cnames = x[0], x[2], x[4], x[5]
            inp,out = modules.get(gt,(None,None))
            # sementara isi class_map=0, nanti di-overwrite oleh structural labels
            class_map[i]=0; membership[i]=netlist
            if out is not None:
                for q,pn in enumerate(pnames):
                    if pn in out:
                        conn = cnames[q]
                        if conn in CONST_NETS: continue
                        for nbr in lookuplist[netlist].get(conn, []):
                            adj[i, nbr+offset]=True; adj[nbr+offset, i]=True
                            if i in train_indices:
                                adj_tr[i, nbr+offset]=True; adj_tr[nbr+offset, i]=True
            i+=1
        netlist+=1
    adj.setdiag(False); adj_tr.setdiag(False)
    num_neighbs = np.asarray(adj.sum(axis=1)).ravel().astype(int)
    return adj.tocsr(), adj_tr.tocsr(), class_map, membership, num_neighbs

# ======== STRUCTURAL HT LABELING ========
def build_node_ios(numnodes, infolist, modules, membership):
    """Kembalikan: types[], names[], in_nets[list], out_nets[list] sebaris index global."""
    types=['']*numnodes; names=['']*numnodes
    in_nets=[[] for _ in range(numnodes)]
    out_nets=[[] for _ in range(numnodes)]
    idx=0
    for net_id, info in enumerate(infolist):
        for x in info:
            gt, inst, pnames, cnames = x[0], x[2], x[4], x[5]
            types[idx]=gt; names[idx]=inst
            inp,out = modules.get(gt,(None,None))
            if inp:
                for q,pn in enumerate(pnames):
                    if pn in inp and cnames[q] not in CONST_NETS:
                        in_nets[idx].append(cnames[q])
            if out:
                for q,pn in enumerate(pnames):
                    if pn in out and cnames[q] not in CONST_NETS:
                        out_nets[idx].append(cnames[q])
            idx+=1
    return types, names, in_nets, out_nets

def fanin_bits_from_same_bus(u, in_nets, hops2_preds, types):
    """Hitung banyaknya bit dari 'bus' yang sama di cone fan-in (~2 hop)."""
    nets=[]
    for p in hops2_preds.get(u, []):
        nets += in_nets[p]
    bases = [re.split(r'[\[\]\._]', s)[0] for s in nets if s]
    return Counter(bases).most_common(1)[0][1] if bases else 0

def structural_labels(adj, infolist, modules, membership):
    n = adj.shape[0]
    types, names, in_nets, out_nets = build_node_ios(n, infolist, modules, membership)
    G = nx.from_scipy_sparse_array(adj)  # undirected

    # precompute 1-2 hop predecessors (approx)
    preds1 = {u: list(G.neighbors(u)) for u in range(n)}
    preds2 = {u: set(preds1[u]) | set([pp for p in preds1[u] for pp in preds1.get(p,[])]) for u in range(n)}

    suspect_trigger=set(); suspect_payload=set(); suspect_sink=set()

    for u in range(n):
        t = types[u]
        # trigger via LUT equality / AND-tree
        if t in PRIMS_LUT:
            if fanin_bits_from_same_bus(u, in_nets, preds2, types) >= 6:
                suspect_trigger.add(u)
        # trigger via counter cluster (FF ~ CARRY4 neighborhood)
        if t in PRIMS_SEQ:
            if any(types[v] in PRIMS_CARRY for v in preds1.get(u, [])):
                suspect_trigger.add(u)
        # sink: output nets dengan nama sensitif
        if any(SINK_PATTERNS.search(nn or '') for nn in out_nets[u]):
            suspect_sink.add(u)

    # payload: MUX/LUT yang menerima input dari trigger dan menuju sink cone
    for u in range(n):
        t = types[u]
        if t in PRIMS_MUX or t in PRIMS_LUT:
            if set(preds1.get(u, [])) & suspect_trigger:
                suspect_payload.add(u)

    # Skoring & label
    labels = np.zeros(n, dtype=int)
    for u in range(n):
        score = 0
        if u in suspect_trigger: score += 1
        if u in suspect_payload: score += 2
        if u in suspect_sink:    score += 2
        # soft-hints dari nama & nets
        blob = " ".join([names[u]] + in_nets[u] + out_nets[u])
        if HT_NAME_RX.search(blob): score += 1
        if USE_ENA_HINT and ENA_RX.search(blob): score += 1
        labels[u] = 1 if score >= 2 else 0
    return labels

# ======== FEATURES ========
def generate_features(numnodes, gatelookup, modules, infolist, priminpslist, primoutslist, membership, num_neighbs):
    featsleft  = np.zeros((numnodes, len(gatelookup)), dtype=np.float32)
    featsright = np.zeros((numnodes, 4), dtype=np.float32)
    connected_to_prim_inp=set(); connected_to_prim_out=set()
    flat_info = [it for sub in infolist for it in sub]
    for i in range(numnodes):
        gt,gtt,inst,label,pnames,cnames = flat_info[i]
        primins, primouts = priminpslist[membership[i]], primoutslist[membership[i]]
        featsleft[i, gatelookup[gt]] = 1.0
        inp_list, out_list = modules.get(gt,(None,None))
        featsright[i,0] = 0 if inp_list is None else len(inp_list)/10.0
        featsright[i,1] = num_neighbs[i]/20.0
        for conn in cnames:
            if conn in primins:  connected_to_prim_inp.add(i);  featsright[i,2]=1
            if conn in primouts: connected_to_prim_out.add(i); featsright[i,3]=1
    feats = np.concatenate((featsleft, featsright), axis=1)
    return feats, connected_to_prim_inp, connected_to_prim_out

def get_neighbs(graph, nodes):
    ret=[]; 
    for i in nodes: ret += list(graph.neighbors(i))
    return ret

def calculate_distances(graph, current, label, output, pos):
    neighbs=set()
    for n in get_neighbs(graph, current):
        if output[n, pos] == -1:
            neighbs.add(n); output[n, pos] = label
    current = list(neighbs)
    if len(current)>0: output = calculate_distances(graph, current, label+1, output, pos)
    return output

def save_output(trainindices, valindices, testindices, adj, adj_train, class_map_onehot, feats):
    role = {'tr': trainindices, 'va': valindices, 'te': testindices}
    save_npz('adj_full.npz', adj); save_npz('adj_train.npz', adj_train)
    with open('class_map.json','w') as fp:
        import json; json.dump(class_map_onehot, fp)
    with open('role.json','w') as fp:
        import json; json.dump(role, fp)
    np.save('feats.npy', feats, allow_pickle=False)

    # pastikan float32
    feats = feats.astype(np.float32, copy=False)
    np.save('feats.npy', feats, allow_pickle=False)

def recordinfo(infolistlist, vlist):
    i=0; testname = os.path.basename(vlist[2][0]) if vlist[2] else 'test'
    with open('logfile_' + testname, 'w') as f: print(testname, file=f)
    for info in infolistlist:
        for l in info:
            with open('logfile_' + testname, 'a') as f:
                print(i, ' : ', l[2], ' : ', l[1], file=f)
            i+=1

# ================== MAIN ==================
def main(input_folder):
    if not input_folder.endswith(os.sep): input_folder += os.sep

    # (1) ambil daftar file
    verilog_paths = getverilogs(input_folder)
    # (2) parsing blok instansiasi
    lines = [readlogs(split, verbose=False) for split in verilog_paths]
    lines = fixassigns(lines)
    # (3) primitif library
    moduleinfo = getmoduleinfo(input_folder)
    modules    = parse_modules(moduleinfo)
    print(f"Jumlah primitive modules : {len(moduleinfo)}")
    print(f"Nama modules primitif    : {sorted(modules.keys())}")

    # (4) primary IO top-level per file
    primin_list, primout_list = getprimlist_from_files(verilog_paths)

    # (5) parse → struktur node/edge dasar
    gates, nodeslist, infolist, indexlist, (train_idx, val_idx, test_idx) = \
        parse_lines(lines, verilog_paths, modules)
    numnodes = sum(len(info) for info in infolist)
    print(f"Gate terdeteksi          : {sorted(gates)}")
    print(f"Total instance count     : {numnodes}")

    # (6) adjacency + placeholder class_map
    lookup = generate_lookup(infolist, modules)
    adj, adj_tr, cmap_placeholder, membership, nneigh = connect(
        numnodes, infolist, lookup, modules, train_idx
    )

    # (7) FEATURES
    gatelookup = {g:i for i,g in enumerate(sorted(gates))}
    feats, c2pi, c2po = generate_features(
        numnodes, gatelookup, modules,
        infolist, primin_list, primout_list,
        membership, nneigh
    )

    # (8) STRUCTURAL LABELS (inti!)
    labels = structural_labels(adj, infolist, modules, membership)

    # (9) one-hot class_map
    class_map_onehot = {str(i): ([1,0] if labels[i]==0 else [0,1]) for i in range(numnodes)}

    # (10) distances (optional features)
    G = nx.from_scipy_sparse_array(adj)
    distances = np.full((numnodes,2), -1.0, dtype=np.float32)
    if len(c2pi)>0: distances[list(c2pi),0] = 0
    if len(c2po)>0: distances[list(c2po),1] = 0
    distances = calculate_distances(G, c2po, 1, distances, 1)
    distances = calculate_distances(G, c2pi, 1, distances, 0)
    distances /= 10.0
    np.savetxt("distances.txt", distances, fmt="%1.1f")
    feats = np.concatenate((feats, distances), axis=1)
    feats = feats.astype(np.float32, copy=False)

    # (11) simpan
    nx.write_gexf(G, "test.gexf")
    save_output(train_idx, val_idx, test_idx, adj, adj_tr, class_map_onehot, feats)

    # -------- Sanity checks --------
    avg_deg = (adj.nnz // 2) / adj.shape[0]
    print("Rata-rata degree:", avg_deg)
    print("Seeds to PI:", len(c2pi), "| Seeds to PO:", len(c2po))
    print("Contoh degree 10 node pertama:", nneigh[:10].tolist())

    # Ringkasan per file
    flat_files = [fp for split in verilog_paths for fp in split]
    assert len(flat_files)==len(infolist), "Mismatch file vs blok info."
    grand_total=0; per_type_grand=Counter(); per_file_summary=[]
    print("\n=== RINGKASAN PER FILE ===")
    for info, fp in zip(infolist, flat_files):
        ctr = Counter(x[0] for x in info); total=sum(ctr.values())
        grand_total += total; per_type_grand.update(ctr)
        per_file_summary.append((os.path.basename(fp), total, ctr))
        top5 = ", ".join(f"{k}:{v}" for k,v in ctr.most_common(5))
        print(f"{os.path.basename(fp):22s}  total={total:6d}  top5= {top5}")
    print("\n=== TOTAL KESELURUHAN ===")
    print("Total instance:", grand_total)
    print("Distribusi tipe (global):", dict(per_type_grand))
    all_types = sorted(set(t for _,_,ctr in per_file_summary for t in ctr.keys()))
    with open("gate_counts_per_file.csv","w", newline="") as fcsv:
        w=csv.writer(fcsv); w.writerow(["file","total"]+all_types)
        for fname,total,ctr in per_file_summary:
            w.writerow([fname,total]+[ctr.get(t,0) for t in all_types])
    print("-> Tersimpan: gate_counts_per_file.csv")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input_folder', default='.', help='Folder dengan subfolder train/, val/, test/')
    # Tambahan argumen mode
    p.add_argument('--mode', choices=['strict','default'], default='strict',
                   help='strict = tanpa hint nama; default = pakai hint trojan|trigger|payload|T\\d+')

    args = p.parse_args()

    # override config bila perlu
    if args.mode == 'default':
        USE_ENA_HINT = False
        HT_NAME_RX   = re.compile(r'trojan|trigger|payload|t\d{2,4}', re.I)


    main(args.input_folder)

