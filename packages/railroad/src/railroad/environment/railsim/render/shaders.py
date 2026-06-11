"""GLSL 330 shader sources.

One program renders to two attachments: flat-shaded color (RGBA8) and
linear view distance in meters (R32F). The distance output replicates the
old Unity ``RenderDist`` semantics without the 24-bit RGB packing.
"""

VERTEX_SHADER = """
#version 330

uniform mat4 u_view_proj;

in vec3 in_position;
in vec3 in_normal;
in vec3 in_color;

out vec3 v_world_pos;
out vec3 v_normal;
out vec3 v_color;

void main() {
    v_world_pos = in_position;
    v_normal = in_normal;
    v_color = in_color;
    gl_Position = u_view_proj * vec4(in_position, 1.0);
}
"""

# Hard ceiling for the light-array size: implementations pad vec3 array
# elements to vec4 (4 components each), and the common fragment-uniform
# budget is 4096 components (macOS GL 4.1, Mesa). The renderer compiles the
# array at the actual light count, so small scenes pay nothing; office-scale
# maps reach ~800 lights and still fit.
MAX_LIGHTS = 960

_FRAGMENT_HEADER = """
#version 330

const int MAX_LIGHTS = {max_lights};
"""

_FRAGMENT_BODY = """
uniform vec3 u_camera_pos;
uniform float u_ambient;
uniform float u_headlight;     // 1.0 = camera-attached light (no-lights mode)
uniform int u_num_lights;
uniform vec3 u_light_positions[MAX_LIGHTS];  // fixture anchors on the ceiling

// Each fixture replicates the Unity light-urp prefab: a downward spot and
// a bare point light hung below the same ceiling anchor.
uniform vec3 u_light_color;       // linear, shared by both lights
uniform float u_spot_drop;        // spot is this far below the anchor
uniform float u_spot_intensity;
uniform float u_spot_range;
uniform vec2 u_spot_cos;          // cos of (inner, outer) half-angles
uniform float u_point_drop;
uniform float u_point_intensity;
uniform float u_point_range;

// Baked shadows for the spots (the point lights are shadowless, matching
// Unity). The atlas holds one tile per light: distance-from-light rendered
// by a camera looking straight down with x/y axes equal to the world's.
uniform float u_shadows;          // 1.0 = shadow atlas is populated
uniform sampler2D u_shadow_atlas;
uniform ivec2 u_shadow_grid;      // atlas tiles (cols, rows)
uniform float u_shadow_tan_half;  // tan of half the bake-camera FOV
uniform float u_shadow_bias;

// Baked shadows for the point lights (off in Unity; optional here for
// realism). Five 90-degree faces per light (+x, -x, +y, -y, down) -- the
// up cone is never occluded, since lights keep >= 2 m of wall clearance
// and hang only 0.8 m below the ceiling.
uniform float u_point_shadows;    // 1.0 = point atlas is populated
uniform sampler2D u_point_atlas;
uniform ivec2 u_point_grid;       // atlas tiles (cols, rows)
uniform float u_point_inset;      // half-texel clamp, keeps uv inside a tile

uniform float u_tonemap;          // 1.0 = URP Neutral tonemapping

in vec3 v_world_pos;
in vec3 v_normal;
in vec3 v_color;

layout(location = 0) out vec4 out_color;
layout(location = 1) out float out_dist;

vec3 srgb_to_linear(vec3 c) { return pow(c, vec3(2.2)); }
vec3 linear_to_srgb(vec3 c) { return pow(c, vec3(1.0 / 2.2)); }

// URP windowed inverse-square distance attenuation.
float distance_att(float d, float range) {
    float w = clamp(1.0 - pow(d / range, 4.0), 0.0, 1.0);
    return w * w / (d * d);
}

// URP Neutral tonemapping (Hable curve, whiteLevel 5.3, whiteClip 1).
vec3 neutral_tonemap(vec3 x) {
    const float a = 0.2, b = 0.29, c = 0.24, d = 0.272, e = 0.02, f = 0.3;
    const float scale = 1.31339;  // 1 / curve(whiteLevel)
    x *= scale;
    x = (x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f) - e / f;
    return x * scale;
}

void main() {
    vec3 to_cam = u_camera_pos - v_world_pos;
    float dist = length(to_cam);
    vec3 n = normalize(v_normal);
    // Two-sided shading: flip the normal toward the viewer so winding and
    // back-faces never render black.
    if (dot(n, to_cam) < 0.0) n = -n;

    vec3 albedo = srgb_to_linear(v_color);
    vec3 light = vec3(u_ambient);
    light += u_headlight * (1.0 - u_ambient)
             * max(dot(n, to_cam / dist), 0.0);

    for (int i = 0; i < u_num_lights; i++) {
        vec3 anchor = u_light_positions[i];

        // Spot, pointing straight down, with URP's squared smooth cone
        // falloff between the inner and outer half-angles.
        vec3 Ls = anchor - vec3(0.0, 0.0, u_spot_drop) - v_world_pos;
        float ds = max(length(Ls), 0.05);
        vec3 ls = Ls / ds;
        float cone = clamp((ls.z - u_spot_cos.y)
                           / (u_spot_cos.x - u_spot_cos.y), 0.0, 1.0);
        float shadow = 1.0;
        if (u_shadows > 0.5 && Ls.z > 0.05) {
            vec2 t = clamp(-Ls.xy / (u_shadow_tan_half * Ls.z),
                           -1.0, 1.0) * 0.5 + 0.5;
            vec2 uv = (vec2(i % u_shadow_grid.x, i / u_shadow_grid.x) + t)
                      / vec2(u_shadow_grid);
            float blocker = textureLod(u_shadow_atlas, uv, 0.0).r;
            // blocker == 0 means the tile is empty along this ray.
            if (blocker > 0.0 && blocker + u_shadow_bias < ds)
                shadow = 0.0;
        }
        light += u_light_color * (u_spot_intensity * max(dot(n, ls), 0.0)
                                  * cone * cone * shadow
                                  * distance_att(ds, u_spot_range));

        // Point light.
        vec3 Lp = anchor - vec3(0.0, 0.0, u_point_drop) - v_world_pos;
        float dp = max(length(Lp), 0.05);
        float pshadow = 1.0;
        if (u_point_shadows > 0.5) {
            // Sample from a point lifted one shadow-texel off the surface
            // (normal-offset): receivers seen at grazing angles in the
            // cube faces (tabletops, walls) would otherwise self-shadow.
            vec3 d = -Lp + n * (2.0 * u_point_inset * dp);
            float ax = abs(d.x), ay = abs(d.y), az = abs(d.z);
            if (!(d.z > 0.0 && az >= ax && az >= ay)) {  // up cone: lit
                int face;
                vec2 st;
                if (az >= ax && az >= ay) {
                    face = 4;  st = d.xy / az;
                } else if (ax >= ay) {
                    face = (d.x > 0.0) ? 0 : 1;
                    st = vec2((d.x > 0.0) ? -d.y : d.y, d.z) / ax;
                } else {
                    face = (d.y > 0.0) ? 2 : 3;
                    st = vec2((d.y > 0.0) ? d.x : -d.x, d.z) / ay;
                }
                vec2 t = clamp(st * 0.5 + 0.5,
                               u_point_inset, 1.0 - u_point_inset);
                int tile = i * 5 + face;
                vec2 uv = (vec2(tile % u_point_grid.x, tile / u_point_grid.x)
                           + t) / vec2(u_point_grid);
                float blocker = textureLod(u_point_atlas, uv, 0.0).r;
                if (blocker > 0.0 && blocker + u_shadow_bias < dp)
                    pshadow = 0.0;
            }
        }
        light += u_light_color * (u_point_intensity
                                  * max(dot(n, Lp / dp), 0.0)
                                  * pshadow
                                  * distance_att(dp, u_point_range));
    }

    vec3 color = albedo * light;
    if (u_tonemap > 0.5) color = neutral_tonemap(color);
    out_color = vec4(linear_to_srgb(clamp(color, 0.0, 1.0)), 1.0);
    out_dist = dist;
}
"""


def fragment_shader(max_lights: int = 1) -> str:
    """Fragment shader source with the light array sized to the scene
    (uniform components are a scarce resource; see `MAX_LIGHTS`)."""
    return _FRAGMENT_HEADER.format(max_lights=max(1, max_lights)) + _FRAGMENT_BODY
