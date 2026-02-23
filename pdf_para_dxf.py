"""
Conversor PDF Vetorial -> DXF (Fidelidade Total)
=================================================
Converte PDFs de redes de distribuicao eletrica para DXF
preservando cores RGB exatas, tracejados reais, preenchimentos
solidos e rotacao de textos.

INSTALACAO:
    pip install pymupdf ezdxf

USO:
    python pdf_para_dxf.py arquivo.pdf
    python pdf_para_dxf.py arquivo.pdf saida.dxf
    python pdf_para_dxf.py arquivo.pdf --todas
    python pdf_para_dxf.py arquivo.pdf --pagina 2
"""

import sys
import os
import argparse
import math

try:
    import fitz
except ImportError:
    print("Instale: pip install pymupdf")
    sys.exit(1)

try:
    import ezdxf
    from ezdxf import rgb2int
except ImportError:
    print("Instale: pip install ezdxf")
    sys.exit(1)


# =============================================================
# UTILIDADES DE COR
# =============================================================
def rgb_float_to_int(r, g, b):
    """Converte RGB float (0.0-1.0) para tupla int (0-255)."""
    return (
        max(0, min(255, int(round(r * 255)))),
        max(0, min(255, int(round(g * 255)))),
        max(0, min(255, int(round(b * 255)))),
    )


def rgb_to_hex(ri, gi, bi):
    """Retorna string hex como '00FF00'."""
    return "{:02X}{:02X}{:02X}".format(ri, gi, bi)


def rgb_to_aci(ri, gi, bi):
    """
    Encontra o ACI (AutoCAD Color Index) mais proximo via distancia euclidiana.
    Garante visibilidade excelente mesmo em viewers que ignoram TrueColor.
    """
    # Cores padrao do AutoCAD (ACI 1-7 + tons comuns)
    # Formato: (R, G, B, ACI)
    ACI_REF = [
        (255, 0, 0, 1),      # Vermelho
        (255, 255, 0, 2),    # Amarelo
        (0, 255, 0, 3),      # Verde
        (0, 255, 255, 4),    # Cyan
        (0, 0, 255, 5),      # Azul
        (255, 0, 255, 6),    # Magenta
        (255, 255, 255, 7),  # Branco (fundo preto)
        (0, 0, 0, 7),        # Preto (fundo branco)
        (128, 128, 128, 8),  # Cinza Escuro
        (192, 192, 192, 9),  # Cinza Claro
        (165, 82, 0, 34),    # Marrom/Aterramento
        (255, 127, 0, 30),   # Laranja
        (0, 128, 255, 150),  # Azul Claro
        (0, 100, 0, 94),     # Verde Escuro
    ]
    
    melhor_dist = float('inf')
    aci_escolhido = 7
    
    for r, g, b, aci in ACI_REF:
        dist = math.sqrt((ri - r)**2 + (gi - g)**2 + (bi - b)**2)
        if dist < melhor_dist:
            melhor_dist = dist
            aci_escolhido = aci
            
    return aci_escolhido


# =============================================================
# LINETYPES - Criacao dinamica a partir dos dash patterns do PDF
# =============================================================
_linetype_cache = {}


def criar_linetype_do_pdf(doc_dxf, dashes_str, escala=1.0):
    """Cria linetype no DXF a partir do dash pattern real do PDF, aplicando escala."""
    if not dashes_str or dashes_str == "[] 0" or dashes_str == "None":
        return "CONTINUOUS"

    # Chave do cache inclui a escala para evitar conflitos
    cache_key = "{}_s{:.4f}".format(dashes_str, escala)
    if cache_key in _linetype_cache:
        return _linetype_cache[cache_key]

    try:
        # Extrai numeros de "[4.61, 4.61] 0" ou similar
        import re
        # Busca tudo que parece numero dentro e fora de colchetes
        num_strs = re.findall(r"[-+]?\d*\.\d+|\d+", dashes_str)
        if not num_strs:
            return "CONTINUOUS"
            
        # O padrao e' o primeiro grupo numerico (lista de dashes)
        if "[" in dashes_str and "]" in dashes_str:
            inside = dashes_str[dashes_str.index("[")+1 : dashes_str.index("]")]
            values = [float(v) for v in re.findall(r"[-+]?\d*\.\d+|\d+", inside)]
        else:
            values = [float(v) for v in num_strs[:-1]] if len(num_strs) > 1 else [float(num_strs[0])]

        if not values:
            return "CONTINUOUS"

        # Aplica a escala nos valores (PDF points -> mm)
        # BUGFIX: Fator 0.5x parece ser o ideal para o visual do CAD em relacao ao PDF
        scaled_values = [max(0.01, v * escala * 0.5) for v in values]

        pattern_elements = []
        total = 0
        for i, v in enumerate(scaled_values):
            if i % 2 == 0:
                pattern_elements.append(v)
                total += v
            else:
                pattern_elements.append(-v)
                total += v

        if not pattern_elements:
            return "CONTINUOUS"

        # Se for apenas um valor [dash], assume gap igual
        if len(pattern_elements) == 1:
            gap = pattern_elements[0]
            pattern_elements.append(-gap)
            total += gap

        dxf_pattern = [total] + pattern_elements

        # Nome unico baseado nos valores escalados
        name_parts = "_".join("{:.2f}".format(abs(v)) for v in scaled_values)
        ltype_name = "PDF_DASH_{}".format(name_parts.replace(".", "p"))

        if ltype_name not in doc_dxf.linetypes:
            doc_dxf.linetypes.add(
                ltype_name,
                pattern=dxf_pattern,
                description="Tracejado PDF escala {:.3f} [{}]".format(
                    escala, ", ".join("{:.2f}".format(v) for v in values)
                ),
            )

        _linetype_cache[cache_key] = ltype_name
        return ltype_name

    except Exception:
        return "CONTINUOUS"


# =============================================================
# BEZIER
# =============================================================
def bezier_cubica(p0, p1, p2, p3, n=20):
    pts = []
    for i in range(n + 1):
        t = i / n
        mt = 1 - t
        x = mt**3*p0[0] + 3*mt**2*t*p1[0] + 3*mt*t**2*p2[0] + t**3*p3[0]
        y = mt**3*p0[1] + 3*mt**2*t*p1[1] + 3*mt*t**2*p2[1] + t**3*p3[1]
        pts.append((x, y))
    return pts


def bezier_quadratica(p0, p1, p2, n=12):
    pts = []
    for i in range(n + 1):
        t = i / n
        mt = 1 - t
        x = mt**2*p0[0] + 2*mt*t*p1[0] + t**2*p2[0]
        y = mt**2*p0[1] + 2*mt*t*p1[1] + t**2*p2[1]
        pts.append((x, y))
    return pts


# =============================================================
# LAYERS
# =============================================================
_layers_criados = set()


def obter_ou_criar_layer(doc_dxf, ri, gi, bi, prefixo=""):
    hex_cor = rgb_to_hex(ri, gi, bi)
    nome = "{}COR_{}".format(prefixo, hex_cor)

    if nome not in _layers_criados:
        aci = rgb_to_aci(ri, gi, bi)
        if nome not in doc_dxf.layers:
            layer = doc_dxf.layers.add(nome)
            layer.color = aci
            layer.true_color = rgb2int((ri, gi, bi))
        _layers_criados.add(nome)

    return nome


def obter_layer_texto(doc_dxf, ri, gi, bi, prefixo=""):
    hex_cor = rgb_to_hex(ri, gi, bi)
    nome = "{}TXT_{}".format(prefixo, hex_cor)

    if nome not in _layers_criados:
        aci = rgb_to_aci(ri, gi, bi)
        if nome not in doc_dxf.layers:
            layer = doc_dxf.layers.add(nome)
            layer.color = aci
            layer.true_color = rgb2int((ri, gi, bi))
        _layers_criados.add(nome)

    return nome


# =============================================================
# LINEWEIGHT
# =============================================================
def converter_lineweight(largura_pdf):
    """Converte largura PDF (pontos) para lineweight DXF (centesimos de mm)."""
    if largura_pdf is None or largura_pdf <= 0:
        return 13

    mm = largura_pdf * 0.3528
    centesimos = int(round(mm * 100))

    valores_validos = [
        0, 5, 9, 13, 15, 18, 20, 25, 30, 35, 40, 50, 53, 60,
        70, 80, 90, 100, 106, 120, 140, 158, 200, 211
    ]
    return min(valores_validos, key=lambda v: abs(v - centesimos))


# =============================================================
# CONVERSAO DE PAGINA
# =============================================================
_bloco_poste_id = 0


def _centro_path(path):
    """Calcula centro (cx, cy) de um path baseado nos items."""
    xs, ys = [], []
    for it in path.get("items", []):
        if not it:
            continue
        if it[0] == "l":
            xs.extend([it[1].x, it[2].x])
            ys.extend([it[1].y, it[2].y])
        elif it[0] == "qu":
            q = it[1]
            xs.extend([q.ul.x, q.ur.x, q.ll.x, q.lr.x])
            ys.extend([q.ul.y, q.ur.y, q.ll.y, q.lr.y])
        elif it[0] == "re":
            r = it[1]
            xs.extend([r.x0, r.x1])
            ys.extend([r.y0, r.y1])
    if not xs:
        return None, None
    return (max(xs)+min(xs))/2, (max(ys)+min(ys))/2


def _tem_quad(path):
    """Verifica se o path tem um item 'qu' (Quad)."""
    for it in path.get("items", []):
        if it and it[0] == "qu":
            return True
    return False


def _eh_fill_circular(path):
    """Verifica se e' um preenchimento circular (>=8 linhas, fill, bbox <5)."""
    items = path.get("items", [])
    fill = path.get("fill")
    if not fill:
        return False
    n_l = sum(1 for it in items if it and it[0] == "l")
    if n_l < 8:
        return False
    xs, ys = [], []
    for it in items:
        if it and it[0] == "l":
            xs.extend([it[1].x, it[2].x])
            ys.extend([it[1].y, it[2].y])
    if not xs:
        return False
    return (max(xs)-min(xs)) < 5 and (max(ys)-min(ys)) < 5


def _eh_cruz_poste(path):
    """Verifica se e' a linha de cruz do poste (1 linha, preto, w~1.15, <12 unidades)."""
    items = path.get("items", [])
    cor = path.get("color")
    if not cor:
        return False
    if len(items) != 1 or items[0][0] != "l":
        return False
    w = path.get("width", 0) or 0
    if w < 0.5:
        return False
    ri, gi, bi = rgb_float_to_int(cor[0], cor[1], cor[2])
    if ri > 10 or gi > 10 or bi > 10:
        return False
    it = items[0]
    bw = abs(it[2].x - it[1].x)
    bh = abs(it[2].y - it[1].y)
    return bw < 12 and bh < 12


def converter_pagina(page, msp, doc_dxf, offset_y=0, escala=1.0, prefixo=""):
    global _bloco_poste_id
    H = page.rect.height

    def tx(x):
        return x * escala

    def ty(y):
        return (H - y) * escala + offset_y

    total = 0
    paths = page.get_drawings()

    # ----------------------------------------------------------
    # FASE 1: Identificar postes (agrupamento por centro)
    # Cada poste e' composto de:
    #   - 1 path com qu (Quad) = quadrado
    #   - 1 path com 1 linha (cruz) no mesmo centro
    #   - 1 path com fill circular (>=8 linhas, fill, pequeno) no mesmo centro
    # ----------------------------------------------------------
    poste_centros = []  # lista de (cx, cy)
    quad_indices = set()

    # Encontra posicoes dos quads (quadrados dos postes)
    for i, p in enumerate(paths):
        if _tem_quad(p):
            # Filtra quads muito pequenos (nao sao postes)
            items = p.get("items", [])
            for it in items:
                if it and it[0] == "qu":
                    q = it[1]
                    bw = max(q.ul.x, q.ur.x, q.ll.x, q.lr.x) - min(q.ul.x, q.ur.x, q.ll.x, q.lr.x)
                    bh = max(q.ul.y, q.ur.y, q.ll.y, q.lr.y) - min(q.ul.y, q.ur.y, q.ll.y, q.lr.y)
                    if bw > 5 and bh > 5:
                        cx = (q.ul.x + q.ur.x + q.ll.x + q.lr.x) / 4
                        cy = (q.ul.y + q.ur.y + q.ll.y + q.lr.y) / 4
                        poste_centros.append((cx, cy, i))
                        quad_indices.add(i)

    # Para cada poste, encontra paths associados
    poste_grupos = {}  # index -> lista de path indices
    idx_usados_por_poste = set()

    for cx, cy, qi in poste_centros:
        grupo = [qi]
        idx_usados_por_poste.add(qi)

        for j, p in enumerate(paths):
            if j == qi or j in idx_usados_por_poste:
                continue

            pcx, pcy = _centro_path(p)
            if pcx is None:
                continue

            dist = math.sqrt((pcx - cx)**2 + (pcy - cy)**2)

            # Cruz: linha unica preta perto do centro
            if dist < 3 and _eh_cruz_poste(p):
                grupo.append(j)
                idx_usados_por_poste.add(j)
                continue

            # Fill circular (bolinha) perto do centro
            if dist < 3 and _eh_fill_circular(p):
                grupo.append(j)
                idx_usados_por_poste.add(j)
                continue

        poste_grupos[qi] = grupo

    # ----------------------------------------------------------
    # FASE 2: Criar BLOCKs para os postes
    # ----------------------------------------------------------
    for qi, grupo in poste_grupos.items():
        # Calcula centro do quad
        quad_path = paths[qi]
        for it in quad_path.get("items", []):
            if it and it[0] == "qu":
                q = it[1]
                cx = (q.ul.x + q.ur.x + q.ll.x + q.lr.x) / 4
                cy = (q.ul.y + q.ur.y + q.ll.y + q.lr.y) / 4
                break

        nome_bloco = "{}POSTE_{}".format(prefixo, _bloco_poste_id)
        _bloco_poste_id += 1

        block = doc_dxf.blocks.new(name=nome_bloco)

        for pi in grupo:
            p = paths[pi]
            cor = p.get("color")
            fill = p.get("fill")
            w = p.get("width") or 0.0

            if cor:
                ri, gi, bi = rgb_float_to_int(cor[0], cor[1], cor[2])
            elif fill:
                ri, gi, bi = rgb_float_to_int(fill[0], fill[1], fill[2])
            else:
                ri, gi, bi = 0, 0, 0

            lw = converter_lineweight(w)

            for it in p.get("items", []):
                if not it:
                    continue

                if it[0] == "l":
                    p1x = (it[1].x - cx) * escala
                    p1y = -(it[1].y - cy) * escala
                    p2x = (it[2].x - cx) * escala
                    p2y = -(it[2].y - cy) * escala
                    block.add_line(
                        (p1x, p1y), (p2x, p2y),
                        dxfattribs={"color": 256, "lineweight": lw}
                    )

                elif it[0] == "qu":
                    qq = it[1]
                    pts = [
                        ((qq.ul.x - cx)*escala, -(qq.ul.y - cy)*escala),
                        ((qq.ur.x - cx)*escala, -(qq.ur.y - cy)*escala),
                        ((qq.lr.x - cx)*escala, -(qq.lr.y - cy)*escala),
                        ((qq.ll.x - cx)*escala, -(qq.ll.y - cy)*escala),
                    ]
                    for k in range(4):
                        block.add_line(pts[k], pts[(k+1)%4],
                            dxfattribs={"color": 256, "lineweight": lw})

            # HATCH solid para paths com fill (bolinha)
            if fill:
                fill_ri, fill_gi, fill_bi = rgb_float_to_int(fill[0], fill[1], fill[2])
                pontos = []
                for it in p.get("items", []):
                    if it and it[0] == "l":
                        px = (it[1].x - cx) * escala
                        py = -(it[1].y - cy) * escala
                        pontos.append((px, py))

                pontos_unicos = []
                seen = set()
                for pt in pontos:
                    key = (round(pt[0], 2), round(pt[1], 2))
                    if key not in seen:
                        seen.add(key)
                        pontos_unicos.append(pt)

                if len(pontos_unicos) >= 3:
                    try:
                        hatch = block.add_hatch(
                            dxfattribs={
                                "color": 256,
                                "true_color": rgb2int((fill_ri, fill_gi, fill_bi)),
                            }
                        )
                        hatch.paths.add_polyline_path(
                            pontos_unicos + [pontos_unicos[0]],
                            is_closed=True
                        )
                        hatch.set_solid_fill()
                    except Exception:
                        pass

        # Insere o bloco no modelspace
        layer = obter_ou_criar_layer(doc_dxf, 0, 0, 0, prefixo)
        msp.add_blockref(
            nome_bloco,
            insert=(tx(cx), ty(cy)),
            dxfattribs={
                "layer": layer,
                "true_color": rgb2int((0, 0, 0)),
            }
        )
        total += 1

    # ----------------------------------------------------------
    # FASE 3: Desenhar paths normais (nao-postes)
    # ----------------------------------------------------------
    for idx, path in enumerate(paths):
        if idx in idx_usados_por_poste:
            continue

        cor = path.get("color")
        fill = path.get("fill")
        dashes = path.get("dashes")
        largura = path.get("width") or 0.0
        items = path.get("items", [])

        if not items:
            continue

        # --- Determina cor do stroke ---
        if cor:
            ri, gi, bi = rgb_float_to_int(cor[0], cor[1], cor[2])
        elif fill and not (fill[0] > 0.95 and fill[1] > 0.95 and fill[2] > 0.95):
            ri, gi, bi = rgb_float_to_int(fill[0], fill[1], fill[2])
        else:
            ri, gi, bi = 0, 0, 0

        layer = obter_ou_criar_layer(doc_dxf, ri, gi, bi, prefixo)

        # --- Linetype ---
        dashes_str = str(dashes) if dashes else ""
        linetype = criar_linetype_do_pdf(doc_dxf, dashes_str, escala=escala)

        # --- Lineweight ---
        lw = converter_lineweight(largura)

        atribs = {
            "layer": layer,
            "color": rgb_to_aci(ri, gi, bi),
            "true_color": rgb2int((ri, gi, bi)),
            "linetype": linetype,
            "lineweight": lw,
        }

        # --- HATCH para fill (preenchimento solido) ---
        if fill:
            fill_ri, fill_gi, fill_bi = rgb_float_to_int(fill[0], fill[1], fill[2])

            pontos = []
            for it in items:
                if not it:
                    continue
                if it[0] == "l":
                    pontos.append((tx(it[1].x), ty(it[1].y)))
                    pontos.append((tx(it[2].x), ty(it[2].y)))
                elif it[0] == "re":
                    rect = it[1]
                    pontos.extend([
                        (tx(rect.x0), ty(rect.y0)),
                        (tx(rect.x1), ty(rect.y0)),
                        (tx(rect.x1), ty(rect.y1)),
                        (tx(rect.x0), ty(rect.y1)),
                    ])

            pontos_unicos = []
            seen = set()
            for p in pontos:
                key = (round(p[0], 2), round(p[1], 2))
                if key not in seen:
                    seen.add(key)
                    pontos_unicos.append(p)

            if len(pontos_unicos) >= 3:
                try:
                    fill_layer = obter_ou_criar_layer(
                        doc_dxf, fill_ri, fill_gi, fill_bi, prefixo
                    )
                    hatch = msp.add_hatch(
                        dxfattribs={
                            "layer": fill_layer,
                            "color": rgb_to_aci(fill_ri, fill_gi, fill_bi),
                            "true_color": rgb2int((fill_ri, fill_gi, fill_bi)),
                        }
                    )
                    hatch.paths.add_polyline_path(
                        pontos_unicos + [pontos_unicos[0]],
                        is_closed=True
                    )
                    hatch.set_solid_fill()
                    total += 1
                except Exception:
                    pass

        # --- Se nao tem stroke, so desenha o fill (ja feito acima) ---
        if not cor:
            continue

        # --- Desenha geometria do stroke ---
        pt_acumulados = []

        def flush_points():
            nonlocal pt_acumulados
            if len(pt_acumulados) >= 2:
                # Remove pontos duplicados consecutivos
                pts_limpos = [pt_acumulados[0]]
                for i in range(1, len(pt_acumulados)):
                    if math.dist(pt_acumulados[i], pts_limpos[-1]) > 0.001:
                        pts_limpos.append(pt_acumulados[i])
                
                if len(pts_limpos) >= 2:
                    msp.add_lwpolyline(pts_limpos, dxfattribs=atribs)
                    return 1
            pt_acumulados = []
            return 0

        for item in items:
            if not item:
                continue
            tipo = item[0]

            try:
                if tipo == "l":
                    p1, p2 = item[1], item[2]
                    v1 = (tx(p1.x), ty(p1.y))
                    v2 = (tx(p2.x), ty(p2.y))
                    
                    if not pt_acumulados:
                        pt_acumulados = [v1, v2]
                    else:
                        # Se o novo ponto inicial for proximo do ultimo ponto acumulado, continua
                        if math.dist(v1, pt_acumulados[-1]) < 0.01:
                            pt_acumulados.append(v2)
                        else:
                            total += flush_points()
                            pt_acumulados = [v1, v2]

                elif tipo == "c":
                    if len(item) < 5:
                        continue
                    pts = bezier_cubica(
                        (tx(item[1].x), ty(item[1].y)),
                        (tx(item[2].x), ty(item[2].y)),
                        (tx(item[3].x), ty(item[3].y)),
                        (tx(item[4].x), ty(item[4].y)),
                    )
                    
                    if not pt_acumulados:
                        pt_acumulados.extend(pts)
                    else:
                        if math.dist(pts[0], pt_acumulados[-1]) < 0.01:
                            pt_acumulados.extend(pts[1:])
                        else:
                            total += flush_points()
                            pt_acumulados.extend(pts)

                elif tipo == "qu":
                    total += flush_points() # Garante que nada acumulado vaze
                    quad = item[1]
                    try:
                        p_ul = (tx(quad.ul.x), ty(quad.ul.y))
                        p_ur = (tx(quad.ur.x), ty(quad.ur.y))
                        p_lr = (tx(quad.lr.x), ty(quad.lr.y))
                        p_ll = (tx(quad.ll.x), ty(quad.ll.y))
                        msp.add_lwpolyline([p_ul, p_ur, p_lr, p_ll, p_ul], dxfattribs=atribs)
                        total += 1
                    except Exception:
                        pass

                elif tipo == "re":
                    total += flush_points()
                    rect = item[1]
                    pts = [
                        (tx(rect.x0), ty(rect.y0)),
                        (tx(rect.x1), ty(rect.y0)),
                        (tx(rect.x1), ty(rect.y1)),
                        (tx(rect.x0), ty(rect.y1)),
                        (tx(rect.x0), ty(rect.y0)),
                    ]
                    msp.add_lwpolyline(pts, dxfattribs=atribs)
                    total += 1

            except Exception:
                pass
        
        # Flush final
        total += flush_points()

    # ----------------------------------------------------------
    # TEXTOS - com rotacao preservada
    # ----------------------------------------------------------
    blocos = page.get_text("dict")

    for bloco in blocos.get("blocks", []):
        if bloco.get("type") != 0:
            continue
        for linha in bloco.get("lines", []):
            # Extrai direcao do texto para calcular rotacao
            direction = linha.get("dir", (1.0, 0.0))
            dx_dir, dy_dir = direction

            # Calcula angulo em graus
            # No PDF: dir = (cos(a), sin(a)) onde a e' o angulo
            # No DXF: rotation e' em graus, sentido anti-horario
            # Mas como invertemos Y (PDF y-down -> DXF y-up), precisamos
            # inverter o sinal de dy
            angulo = math.degrees(math.atan2(-dy_dir, dx_dir))

            for span in linha.get("spans", []):
                texto_bruto = span.get("text", "")
                if not texto_bruto.strip():
                    continue

                size = span.get("size", 5)
                if size < 0.3:
                    continue

                origin = span.get("origin")
                if origin:
                    x0, y0 = tx(origin[0]), ty(origin[1])
                else:
                    bbox = span["bbox"]
                    x0, y0 = tx(bbox[0]), ty(bbox[3])

                # Cor do texto
                cor_int = span.get("color", 0)
                if cor_int == 0 or cor_int is None:
                    tri, tgi, tbi = 0, 0, 0
                else:
                    tri = (cor_int >> 16) & 0xFF
                    tgi = (cor_int >> 8) & 0xFF
                    tbi = cor_int & 0xFF

                layer_txt = obter_layer_texto(doc_dxf, tri, tgi, tbi, prefixo)
                altura_txt = size * escala

                try:
                    # Revertendo para MTEXT para garantia de TrueColor e visibilidade
                    mtext = msp.add_mtext(
                        texto_bruto,
                        dxfattribs={
                            "layer": layer_txt,
                            "char_height": altura_txt,
                            "color": rgb_to_aci(tri, tgi, tbi),
                            "true_color": rgb2int((tri, tgi, tbi)),
                            "insert": (x0, y0),
                            "rotation": angulo,
                        }
                    )
                    mtext.dxf.attachment_point = 7  # bottom-left ( baseline-ish)
                    total += 1
                except Exception:
                    pass

    return total


# =============================================================
# FUNCAO PRINCIPAL
# =============================================================
def converter_pdf_para_dxf(caminho_pdf, caminho_dxf=None,
                            paginas=None, escala=0.3528, versao_dxf="R2000"):
    if not os.path.exists(caminho_pdf):
        print("Arquivo nao encontrado: {}".format(caminho_pdf))
        sys.exit(1)

    if caminho_dxf is None:
        caminho_dxf = os.path.splitext(caminho_pdf)[0] + ".dwg"

    print("\nAbrindo: {}".format(caminho_pdf))
    doc_pdf = fitz.open(caminho_pdf)
    n_pags = len(doc_pdf)
    print("Paginas encontradas: {}".format(n_pags))

    if paginas is None:
        paginas = list(range(n_pags))
    else:
        paginas = [p - 1 for p in paginas if 1 <= p <= n_pags]

    doc_dxf = ezdxf.new(versao_dxf)
    doc_dxf.header["$INSUNITS"] = 0
    doc_dxf.header["$LTSCALE"] = 1.0
    doc_dxf.header["$CELTSCALE"] = 1.0
    msp = doc_dxf.modelspace()

    # Reset caches globais
    global _linetype_cache, _layers_criados, _bloco_poste_id
    _linetype_cache = {}
    _layers_criados = set()
    _bloco_poste_id = 0

    total = 0
    offset_y = 0

    for num in paginas:
        page = doc_pdf[num]
        prefixo = "P{}_".format(num + 1) if len(paginas) > 1 else ""
        print("\nConvertendo pagina {}...".format(num + 1))
        n = converter_pagina(page, msp, doc_dxf,
                             offset_y=offset_y, escala=escala, prefixo=prefixo)
        print("  {} elementos convertidos".format(n))
        total += n
        offset_y -= (page.rect.height * escala) + 50

    doc_dxf.saveas(caminho_dxf)
    doc_pdf.close()

    kb = os.path.getsize(caminho_dxf) / 1024
    print("\n" + "=" * 50)
    print("Conversao concluida!")
    print("  Total de elementos : {}".format(total))
    print("  Arquivo DXF gerado : {}".format(caminho_dxf))
    print("  Tamanho            : {:.1f} KB".format(kb))

    print("\nLayers criados:")
    for layer in doc_dxf.layers:
        if layer.dxf.name != "0" and layer.dxf.name != "Defpoints":
            print("  {}".format(layer.dxf.name))

    lts = [lt.dxf.name for lt in doc_dxf.linetypes
           if lt.dxf.name not in ("Standard", "ByBlock", "ByLayer",
                                   "CONTINUOUS", "Continuous")]
    if lts:
        print("\nLinetypes:")
        for lt in lts:
            print("  {}".format(lt))

    print("=" * 50)


# =============================================================
# CLI
# =============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converte PDF de rede eletrica para DXF (fidelidade total)"
    )
    parser.add_argument("entrada", nargs="?", default="arquivo.pdf",
                        help="Caminho do arquivo PDF de entrada")
    parser.add_argument("saida", nargs="?", default=None,
                        help="Caminho do arquivo DXF de saida (opcional)")
    parser.add_argument("--pagina", "-p", type=int, default=None,
                        help="Numero da pagina a converter (1-based)")
    parser.add_argument("--todas", "-t", action="store_true",
                        help="Converte todas as paginas")
    parser.add_argument("--escala", "-e", type=float, default=0.3528,
                        help="Fator de escala (default: 0.3528 = pontos PDF para mm)")
    parser.add_argument("--versao", default="R2000",
                        choices=["R2000", "R2010", "R2013", "R2018"],
                        help="Versao do DXF (default: R2000)")
    args = parser.parse_args()

    if args.todas:
        paginas = None
    elif args.pagina:
        paginas = [args.pagina]
    else:
        paginas = [1]

    converter_pdf_para_dxf(
        caminho_pdf=args.entrada,
        caminho_dxf=args.saida,
        paginas=paginas,
        escala=args.escala,
        versao_dxf=args.versao
    )