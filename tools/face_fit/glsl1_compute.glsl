// glsl1 -- GNM morph + skinning, anatomical colors and emotion recolor.
// Installed by install_face_tracking.py.
//
// sBasis   636 morph deltas (.xyz, 5 rows per channel) + 4 skin weights;
//          .w of rows 0..4 = region, rows 5..9 = sub-region weight
//          (see region_masks.py)
// sWeights face_fit's output as samples: 636 coefficients, 7 pose channels,
//          6 emotion scores, dominant_id
// sXforms  4 joint matrices (3x4 each)
// uRecolor, uColors, uLipAmount, uLipFeather  Emotion Recolor, Enable Colors,
//          Lip Amount, Lip Feather (floats; .x reads them)
// uNeutral..uFear  Emotion Colors page; uSkin..uTongue  Skin Colors page

const int DOMINANT = 636 + 7 + 6;

float M(int i) { return texelFetch(sXforms, ivec2(i, 0), 0).r; }

vec3 emotionColor(int e) {
    if (e == 1) return uHappy.rgb;
    if (e == 2) return uSurprise.rgb;
    if (e == 3) return uAngry.rgb;
    if (e == 4) return uSad.rgb;
    if (e == 5) return uFear.rgb;
    return uNeutral.rgb;
}

void main() {
    const uint id = TDIndex();
    if (id >= TDNumElements())
        return;
    vec3 p = TDIn_P();
    int i = int(id);
    ivec2 tile = ivec2(i % 4096, i / 4096);
    float region = round(texelFetch(sBasis, tile, 0).w);
    float sub = texelFetch(sBasis, ivec2(tile.x, 5 + tile.y), 0).w;
    bool colors = uColors.x > 0.5;

    for (int k = 0; k < 636; k++) {
        float w = texelFetch(sWeights, ivec2(k, 0), 0).r;
        p += w * texelFetch(sBasis, ivec2(tile.x, k * 5 + tile.y), 0).xyz;
    }
    // the opaque cornea dome would hide the coloured iris: tuck its front in
    if (colors && region == 4.0 && sub < 0.0065)
        p.z -= 0.004;

    vec4 sw = texelFetch(sBasis, ivec2(tile.x, 3180 + tile.y), 0);
    vec4 hp = vec4(p, 1.0);
    vec3 outp = vec3(0.0);
    for (int j = 0; j < 4; j++) {
        int b = j * 12;
        outp += sw[j] * vec3(
            dot(vec4(M(b+0), M(b+1), M(b+2),  M(b+3)),  hp),
            dot(vec4(M(b+4), M(b+5), M(b+6),  M(b+7)),  hp),
            dot(vec4(M(b+8), M(b+9), M(b+10), M(b+11)), hp));
    }
    P[id] = outp;

    // skin is the emotion color when recoloring; without Colors it's everything
    vec3 skin = colors ? uSkin.rgb : vec3(1.0);
    if (uRecolor.x > 0.5)
        skin = emotionColor(int(round(texelFetch(sWeights, ivec2(DOMINANT, 0), 0).r)));
    vec3 col = skin;
    if (colors) {
        if (region == 0.0) {
            // sub 0.5 is the lip border; Lip Feather softens it
            float halfW = max(0.01, uLipFeather.x * 0.45);
            float lip = smoothstep(0.5 - halfW, 0.5 + halfW, sub);
            col = mix(skin, uLip.rgb, lip * uLipAmount.x);
        } else if (region == 1.0) {
            col = mix(uGums.rgb, uTeeth.rgb, sub);
        } else if (region == 2.0) {
            col = uTongue.rgb;
        } else if (region == 3.0 || region == 4.0) {
            float pupil = 1.0 - smoothstep(0.0015, 0.0021, sub);
            float iris = 1.0 - smoothstep(0.0053, 0.0060, sub);
            float limbus = smoothstep(0.0044, 0.0054, sub) * iris;
            vec3 eye = mix(uSclera.rgb, mix(uIris.rgb, uIris.rgb * 0.45, limbus), iris);
            col = mix(eye, uPupil.rgb, pupil);
        } else if (region == 5.0) {
            col = mix(uSkin.rgb, uTongue.rgb * 0.25, sub);   // dark inside the mouth
        }
    }
    Color[id] = vec4(col, 1.0);
}
