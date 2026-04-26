/*
 *  Copyright 2026 Will Zakielarz
 */

#ifndef _wmz_blend_h_
#define _wmz_blend_h_

#include "include/GPixel.h"
#include "include/GBlendMode.h"
#include "include/GMath.h"
#include <algorithm>

// Helper: Blend premul src + dest following GBlendMode
static inline GPixel apply_blend(GPixel src, GPixel dest, GBlendMode blend_mode) {
    // Premultiplied components
    int sa = GPixel_GetA(src);
    int sr = GPixel_GetR(src);
    int sg = GPixel_GetG(src);
    int sb = GPixel_GetB(src);

    int da = GPixel_GetA(dest);
    int dr = GPixel_GetR(dest);
    int dg = GPixel_GetG(dest);
    int db = GPixel_GetB(dest);

    // Predefine for ease
    int a, r, g, b;
    float sa_norm = sa / 255.0f;
    float da_norm = da / 255.0f;
    float one_minus_sa = 1.0f - sa_norm;
    float one_minus_da = 1.0f - da_norm;
    // Blend these colors man!
    switch (blend_mode) {
        // 0
        case GBlendMode::kClear:
            a = 0;
            r = 0;
            g = 0;
            b = 0;
            break;
        // S
        case GBlendMode::kSrc:
            a = sa;
            r = sr;
            g = sg;
            b = sb;
            break;
        // D
        case GBlendMode::kDst:
            a = da;
            r = dr;
            g = dg;
            b = db;
            break;
        // S + (1 - Sa)*D
        case GBlendMode::kSrcOver:
            a = GRoundToInt(sa + one_minus_sa * da);
            r = GRoundToInt(sr + one_minus_sa * dr);
            g = GRoundToInt(sg + one_minus_sa * dg);
            b = GRoundToInt(sb + one_minus_sa * db);
            break;
        // D + (1 - Da)*S
        case GBlendMode::kDstOver:
            a = GRoundToInt(da + one_minus_da * sa);
            r = GRoundToInt(dr + one_minus_da * sr);
            g = GRoundToInt(dg + one_minus_da * sg);
            b = GRoundToInt(db + one_minus_da * sb);
            break;
        // Da * S
        case GBlendMode::kSrcIn:
            a = GRoundToInt(da_norm * sa);
            r = GRoundToInt(da_norm * sr);
            g = GRoundToInt(da_norm * sg);
            b = GRoundToInt(da_norm * sb);
            break;
        // Sa * D
        case GBlendMode::kDstIn:
            a = GRoundToInt(sa_norm * da);
            r = GRoundToInt(sa_norm * dr);
            g = GRoundToInt(sa_norm * dg);
            b = GRoundToInt(sa_norm * db);
            break;
        // (1 - Da) * S
        case GBlendMode::kSrcOut:
            a = GRoundToInt(one_minus_da * sa);
            r = GRoundToInt(one_minus_da * sr);
            g = GRoundToInt(one_minus_da * sg);
            b = GRoundToInt(one_minus_da * sb);
            break;
        // (1 - Sa) * D
        case GBlendMode::kDstOut:
            a = GRoundToInt(one_minus_sa * da);
            r = GRoundToInt(one_minus_sa * dr);
            g = GRoundToInt(one_minus_sa * dg);
            b = GRoundToInt(one_minus_sa * db);
            break;
        // Da*S + (1 - Sa)*D
        case GBlendMode::kSrcATop:
            a = GRoundToInt(da_norm * sa + one_minus_sa * da);
            r = GRoundToInt(da_norm * sr + one_minus_sa * dr);
            g = GRoundToInt(da_norm * sg + one_minus_sa * dg);
            b = GRoundToInt(da_norm * sb + one_minus_sa * db);
            break;
        // Sa*D + (1 - Da)*S
        case GBlendMode::kDstATop:
            a = GRoundToInt(sa_norm * da + one_minus_da * sa);
            r = GRoundToInt(sa_norm * dr + one_minus_da * sr);
            g = GRoundToInt(sa_norm * dg + one_minus_da * sg);
            b = GRoundToInt(sa_norm * db + one_minus_da * sb);
            break;
        // (1 - Sa)*D + (1 - Da)*S
        case GBlendMode::kXor:
            a = GRoundToInt(one_minus_sa * da + one_minus_da * sa);
            r = GRoundToInt(one_minus_sa * dr + one_minus_da * sr);
            g = GRoundToInt(one_minus_sa * dg + one_minus_da * sg);
            b = GRoundToInt(one_minus_sa * db + one_minus_da * sb);
            break;
        // S*D
        case GBlendMode::kModulate:
            a = GRoundToInt(sa * da / 255.0f);
            r = GRoundToInt(sr * dr / 255.0f);
            g = GRoundToInt(sg * dg / 255.0f);
            b = GRoundToInt(sb * db / 255.0f);
            break;
        // S + D - S*D
        case GBlendMode::kScreen:
            a = GRoundToInt(sa + da - sa * da / 255.0f);
            r = GRoundToInt(sr + dr - sr * dr / 255.0f);
            g = GRoundToInt(sg + dg - sg * dg / 255.0f);
            b = GRoundToInt(sb + db - sb * db / 255.0f);
            break;
        // Rc = S + D - max(S*Da, D*Sa), Ra = kSrcOver
        case GBlendMode::kDarken:
            a = GRoundToInt(sa + one_minus_sa * da);
            r = GRoundToInt(sr + dr - std::max(sr * da_norm, dr * sa_norm));
            g = GRoundToInt(sg + dg - std::max(sg * da_norm, dg * sa_norm));
            b = GRoundToInt(sb + db - std::max(sb * da_norm, db * sa_norm));
            break;
        // Rc = S + D - min(S*Da, D*Sa), Ra = kSrcOver
        case GBlendMode::kLighten:
            a = GRoundToInt(sa + one_minus_sa * da);
            r = GRoundToInt(sr + dr - std::min(sr * da_norm, dr * sa_norm));
            g = GRoundToInt(sg + dg - std::min(sg * da_norm, dg * sa_norm));
            b = GRoundToInt(sb + db - std::min(sb * da_norm, db * sa_norm));
            break;
    }

    // Clamp these vals
    a = std::max(0, std::min(255, a));
    r = std::max(0, std::min(255, r));
    g = std::max(0, std::min(255, g));
    b = std::max(0, std::min(255, b));
    return GPixel_PackARGB(a, r, g, b);
}

#endif
