use super::*;

#[test]
fn the_decals_unprojection_rows_are_found() {
    let decal = "void main() {\n\tvec2 window = screenPosToTexcoord( gl_FragCoord.xy, _fa_[1 ] );\n\tvec4 wclip = vec4( window, viewZ, 1.0 );\n\tvec4 worldPos;\n\tworldPos.x = dot( wclip, _fa_[2 ] );\n\tworldPos.y = dot( wclip, _fa_[3 ] );\n\tworldPos.z = dot( wclip, _fa_[4 ] );\n\tworldPos.w = dot( wclip, _fa_[5 ] );\n}\n";
    assert_eq!(unprojection_row(decal), Some(2));
    assert_eq!(unprojection_row(&decal.replace("_fa_[4 ]", "_fa_[9 ]")), None);
    assert_eq!(unprojection_row("void main() { out_FragColor0 = vec4( 1.0 ); }"), None);
}

#[test]
fn depth_of_field_and_grain_are_left_out_of_the_post_process_only() {
    let post = "void main() {\n\tfloat dof;\n\t{\n\t\tdof = max(a, b);\n\t};\n\tfloat blur = 0.0;\n\t{\n\t\tgrain *= tempLum;\n\t\tfinal.xyz += grain;\n\t};\n}\n";
    let out = without_blur_or_grain("postprocess", post).expect("the post-process is known");
    assert!(out.contains("\tdof = ( dof == 3.0e38 ) ? dof : 0.0;\n\tfloat blur = 0.0;"), "{out}");
    assert!(out.contains("final.xyz += ( grain.x == 3.0e38 ) ? grain : vec3( 0.0 );"), "{out}");
    assert!(!out.contains("final.xyz += grain;"), "{out}");
    let values = without_blur_or_grain("calcdepthoffieldvalues", "\tout_FragColor0 = vec4( dof * 0.1 );\n").unwrap();
    assert!(values.contains("vec4( ( dof == 3.0e38 ) ? dof * 0.1 : 0.0 )"), "{values}");
    let kernel = without_blur_or_grain("calcdepthoffieldmaxkernel", "\tout_FragColor0 = vec4( res );\n").unwrap();
    assert!(kernel.contains("vec4( ( res == 3.0e38 ) ? res : 0.0 )"), "{kernel}");
    // Another program, or a text not the one known, is left alone.
    assert_eq!(without_blur_or_grain("decalblood", post), None);
    assert_eq!(without_blur_or_grain("postprocess", "void main() {}"), None);
}

const VERTEX: &str = "#version 150\n\nuniform vec4 _va_ [6];\nuniform vec4 hardwareSkinning ;\nuniform matrices_ubo { vec4 matrices[768]; };\nin vec4 in_Position;\n\nvoid main() {\n\tvec4 position = in_Position * _va_[0 ] + _va_[1 ];\n\tgl_Position.x = dot4( position, _va_[2 ] );\n\tgl_Position.y = dot4( position, _va_[3 ] );\n\tgl_Position.z = dot4( position, _va_[4 ] );\n\tgl_Position.w = dot4( position, _va_[5 ] );\n}\n";

#[test]
fn the_rows_that_give_the_position_are_found() {
    assert_eq!(position_rows(VERTEX), Some([2, 3, 4, 5]));
    assert_eq!(position_rows("void main() {\n\tgl_Position = in_Position;\n}\n"), None);
    assert_eq!(position_rows("\tgl_Position.x = dot( in_Position, _va_[0 ] );\n\tgl_Position.y = dot( in_Position, _va_[1 ] );\n\tgl_Position.z = dot( in_Position, _va_[2 ] );\n\tgl_Position.w = dot( in_Position, _va_[3 ] );\n"), Some([0, 1, 2, 3]));
    // A row scaled afterwards is not a plain row.
    assert_eq!(position_rows("\tgl_Position.x = dot4( in_Position, _va_[0 ] ) * _va_[1 ].x;\n"), None);
}

#[test]
fn plain_uniforms_are_read_and_blocks_left_alone() {
    let found: Vec<Uniform> = uniforms(VERTEX).into_iter().map(|(_, u)| u).collect();
    assert_eq!(
        found,
        vec![
            Uniform { kind: "vec4".into(), name: "_va_".into(), count: 6 },
            Uniform { kind: "vec4".into(), name: "hardwareSkinning".into(), count: 1 },
        ]
    );
    assert_eq!(uniform("uniform sampler2D samp_diffusemap ;"), Some(Uniform { kind: "sampler2D".into(), name: "samp_diffusemap".into(), count: 1 }));
}

#[test]
fn uniforms_take_the_engines_locations() {
    let out = with_locations(VERTEX, &|u| match u.name.as_str() {
        "_va_" => Some(3),
        _ => None,
    });
    assert!(out.starts_with("#version 150\n#extension GL_ARB_explicit_uniform_location : require\n"));
    assert!(out.contains("\nlayout(location = 3) uniform vec4 _va_ [6];\n"));
    assert!(out.contains("\nuniform vec4 hardwareSkinning ;\n"));
    assert_eq!(with_locations(VERTEX, &|_| None), VERTEX);
}

#[test]
fn the_stereo_vertex_shader_keeps_main_and_adds_the_other_eye() {
    let out = stereo_vertex(VERTEX).unwrap();
    assert!(out.starts_with("#version 150\n#extension GL_NV_viewport_array2 : require\n#extension GL_NV_stereo_view_rendering : require\n#extension GL_ARB_gpu_shader5 : require\n"));
    assert!(out.contains("precise vec4 monaka_right"), "the right eye's position worked out alike in every program");
    assert!(out.contains("void monaka_main() {\n\tvec4 position"));
    assert_eq!(out.matches("void main()").count(), 1);
    assert!(out.contains("layout( secondary_view_offset = 1 ) out int gl_Layer;"));
    assert!(out.contains("gl_SecondaryViewportMaskNV[0] = 1;"));
    assert!(out.contains("uniform vec4 monaka_sl[5];"));
    assert_eq!(stereo_vertex("#version 150\nvoid mainly() {}\n"), None);
}

#[test]
fn fragment_outputs_are_named() {
    assert_eq!(outputs("out vec4 out_FragColor0;\nout vec4 out_FragColor1;\nin vec4 vofi;\n"), vec!["out_FragColor0", "out_FragColor1"]);
}

#[test]
fn programs_reading_the_scene_at_their_pixel_are_told_apart() {
    // A decal placing itself by the depth under it.
    assert!(reads_screen("uniform sampler2D samp_viewdepthmap;\nin vec4 gl_FragCoord;\nvoid main() {\n\tvec2 tc = screenPosToTexcoord( gl_FragCoord.xy, _fa_[0 ] );\n\tfloat z = tex2Ddepth( samp_viewdepthmap, tc );\n}\n"));
    // A dither by the pixel, or the scene's map read without it, is the same in either eye's copy.
    assert!(!reads_screen("uniform sampler2D samp_dither256map;\nin vec4 gl_FragCoord;\nvoid main() {\n\tvec2 texcoord = gl_FragCoord.xy * ( 1.0 / 256.0 );\n}\n"));
    assert!(!reads_screen("uniform sampler2D samp_viewcolormap;\nin vec4 vofi_TexCoord0;\nvoid main() {\n}\n"));
}

const LIGHT: &str = "#version 150\n\
vec4 tex2D( sampler2D sampler, vec2 texcoord ) { return texture( sampler, texcoord.xy ); }\n\
vec4 tex2D( sampler2DShadow sampler, vec3 texcoord ) { return vec4( texture( sampler, texcoord.xyz ) ); }\n\
vec4 tex2Dproj( sampler2D sampler, vec3 texcoord ) { return textureProj( sampler, texcoord ); }\n\
float tex2Ddepth ( sampler2D image, vec2 texcoord ) { return tex2D( image, texcoord ).x; }\n\
float ComputeSoftParticleScale ( sampler2D image, vec2 tc, float z, float softness ) {\n\
\tfloat depth = tex2Ddepth( image, tc );\n\
\treturn ( depth - z ) * softness;\n\
}\n\
vec2 screenPosToTexcoord ( vec2 pos, vec4 bias_scale ) { return ( pos * bias_scale.zw + bias_scale.xy ); }\n\
uniform vec4 _fa_ [3];\n\
uniform sampler2D samp_viewdepthmap;\n\
uniform sampler2D samp_lightfalloffmap;\n\
in vec4 gl_FragCoord;\n\
out vec4 out_FragColor0;\n\
void main() {\n\
\tvec2 tc = screenPosToTexcoord( gl_FragCoord.xy, _fa_[0 ] );\n\
\tfloat z = tex2Ddepth( samp_viewdepthmap, tc );\n\
\tfloat s = ComputeSoftParticleScale( samp_viewdepthmap, tc, z, 2.0 );\n\
\tout_FragColor0 = tex2D( samp_lightfalloffmap, vec2( z, 0.5 ) ) * _fa_[ 1 + 1 ] * s;\n\
}\n";

#[test]
fn screen_maps_are_read_at_the_drawn_layer() {
    let layered = layered_fragment(LIGHT).expect("converts");
    assert_eq!(layered.maps, vec!["samp_viewdepthmap"]);
    assert_eq!(layered.rows, 3);
    let s = &layered.source;
    assert!(s.starts_with("#version 150\n#extension GL_ARB_fragment_layer_viewport : require\n"), "{s}");
    // Overloads of the helpers taking a 2D sampler, the lookups at the layer.
    assert!(s.contains("vec4 tex2D( sampler2DArray sampler, vec2 texcoord ) { return texture( sampler, vec3( texcoord.xy, float( gl_Layer ) ) ); }"), "{s}");
    assert!(s.contains("float tex2Ddepth ( sampler2DArray image, vec2 texcoord ) { return tex2D( image, texcoord ).x; }"), "{s}");
    assert!(s.contains("float ComputeSoftParticleScale ( sampler2DArray image, vec2 tc, float z, float softness ) {\n\tfloat depth = tex2Ddepth( image, tc );"), "{s}");
    // Not the projective lookup (arrays have none), nor the shadow sampler's.
    assert!(!s.contains("tex2Dproj( sampler2DArray"), "{s}");
    assert!(!s.contains("sampler2DArrayShadow"), "{s}");
    // The map's calls choose by layer; other maps are left as they are.
    assert!(s.contains("float z = ( gl_Layer != 0 ? tex2Ddepth( monaka_l_samp_viewdepthmap, tc ) : tex2Ddepth( samp_viewdepthmap, tc ) );"), "{s}");
    assert!(s.contains("( gl_Layer != 0 ? ComputeSoftParticleScale( monaka_l_samp_viewdepthmap, tc, z, 2.0 ) : ComputeSoftParticleScale( samp_viewdepthmap, tc, z, 2.0 ) )"), "{s}");
    assert!(s.contains("tex2D( samp_lightfalloffmap, vec2( z, 0.5 ) ) * monaka_fa[ 1 + 1 ]"), "{s}");
    assert!(s.contains("uniform sampler2D samp_viewdepthmap;\nuniform sampler2DArray monaka_l_samp_viewdepthmap;\n"), "{s}");
    assert!(!s.contains("monaka_l_samp_lightfalloffmap"), "{s}");
    // The rows: the other eye's in layer 1.
    assert!(s.contains("uniform vec4 _fa_ [3];\nuniform vec4 monaka_fr[3];\nvec4 monaka_fa[3];\n"), "{s}");
    assert!(s.contains("screenPosToTexcoord( gl_FragCoord.xy, monaka_fa[0 ] )"), "{s}");
    assert!(s.contains("void monaka_fmain() {"), "{s}");
    assert!(s.contains("monaka_fa[i] = gl_Layer != 0 ? monaka_fr[i] : _fa_[i];"), "{s}");
    assert_eq!(uniforms(s).iter().filter(|(_, u)| u.name == "_fa_").count(), 1);
}

#[test]
fn a_screen_map_used_otherwise_is_not_layered() {
    // Passed to a lookup straight: no helper to overload.
    assert_eq!(layered_fragment("#version 150\nuniform sampler2D samp_viewdepthmap;\nin vec4 gl_FragCoord;\nvoid main() {\n\tfloat z = texture( samp_viewdepthmap, gl_FragCoord.xy ).x;\n}\n"), None);
    // A multisampled map is not a 2D one.
    assert_eq!(layered_fragment("#version 150\nuniform sampler2DMS samp_viewtargetms;\nin vec4 gl_FragCoord;\nvoid main() {\n}\n"), None);
}

/// Every render program extracted into `MONAKA_RENDERPROGS` (`tools/sp/resources.py`) that reads
/// the scene's maps, converted.
#[test]
#[ignore]
fn the_games_screen_programs_convert() {
    let Ok(folder) = std::env::var("MONAKA_RENDERPROGS") else { return };
    let (mut reading, mut layered) = (0, 0);
    for entry in std::fs::read_dir(folder).expect("a folder") {
        let bytes = std::fs::read(entry.expect("an entry").path()).expect("readable");
        let text: String = bytes.iter().map(|&b| b as char).collect();
        for run in text.split(|c: char| !(c == '\t' || c == '\n' || c == '\r' || (' '..='~').contains(&c))).filter(|r| r.len() >= 200) {
            if run.contains("gl_Position") || !reads_screen(run) {
                continue;
            }
            reading += 1;
            match layered_fragment(run) {
                Some(l) => {
                    layered += 1;
                    assert!(!l.source.contains("( samp_view") || l.source.contains("monaka_l_"), "{}", l.source);
                }
                None => println!("not layered:\n{}", &run[run.find("void main").unwrap_or(0)..]),
            }
        }
    }
    println!("{layered} of {reading} layered");
}

const RAY_VERTEX: &str = "#version 150\nuniform vec4 _va_ [12];\nout vec4 vofi_TexCoord4;\nvoid main() {\n\tvec4 localPosition = vec4( 1.0 );\n\tgl_Position.x = dot4( localPosition, _va_[4 ] );\n\tgl_Position.y = dot4( localPosition, _va_[5 ] );\n\tgl_Position.z = dot4( localPosition, _va_[6 ] );\n\tgl_Position.w = dot4( localPosition, _va_[7 ] );\n\t\tvofi_TexCoord4.x = dot4( localPosition, _va_[8 ] );\n\t\tvofi_TexCoord4.y = dot4( localPosition, _va_[9 ] );\n\t\tvofi_TexCoord4.z = dot4( localPosition, _va_[10 ] );\n\t\tvofi_TexCoord4.xyz -= _va_[11 ].xyz;\n\t\tvofi_TexCoord4.w = 0.0;\n}\n";
const RAY_FRAGMENT: &str = "#version 150\nin vec4 vofi_TexCoord4;\nuniform vec4 _fa_ [16];\nvoid main() {\n\tvec3 a = normalize( vofi_TexCoord4.xyz );\n\tvec3 b = normalize( -vofi_TexCoord4.xyz.xyz );\n\tvec3 c = normalize( vofi_TexCoord4.xyz + _fa_[10 ].xyz - _fa_[11 ].xyz );\n}\n";

#[test]
fn a_ray_from_the_eye_is_found_by_its_rows_and_origin() {
    assert_eq!(view_rays(RAY_VERTEX), vec![ViewRay { varying: "vofi_TexCoord4".into(), rows: [8, 9, 10] }]);
    assert!(view_rays(VERTEX).is_empty());
}

#[test]
fn the_other_eyes_copy_moves_the_ray_where_it_is_a_direction_only() {
    let stereo = stereo_vertex(RAY_VERTEX).unwrap();
    let (vertex, fragment) = eye_rays(RAY_VERTEX, &stereo, RAY_FRAGMENT).unwrap();
    assert!(vertex.contains("flat out vec3 monaka_eye_vofi_TexCoord4;"));
    assert!(vertex.contains("\tmonaka_main();\n\tmonaka_rays();\n"));
    assert!(vertex.contains("dot( _va_[8].xyz, monaka_apart )"));
    assert!(vertex.contains("monaka_origin( _va_[4], _va_[5], _va_[7] )"));
    let shift = "( gl_Layer == 1 ? monaka_eye_vofi_TexCoord4 : vec3( 0.0 ) )";
    assert!(fragment.contains(&format!("normalize( ( vofi_TexCoord4.xyz + {shift} ) )")));
    assert!(fragment.contains(&format!("normalize( -( vofi_TexCoord4.xyz + {shift} ) )")));
    assert!(fragment.contains("normalize( vofi_TexCoord4.xyz + _fa_[10 ].xyz - _fa_[11 ].xyz )"));
    assert!(fragment.contains("flat in vec3 monaka_eye_vofi_TexCoord4;"));
    assert!(fragment.contains("GL_ARB_fragment_layer_viewport"));
    assert_eq!(eye_rays(RAY_VERTEX, &stereo, "#version 150\nin vec4 vofi_TexCoord4;\nvoid main() {}\n"), None);
}

#[test]
fn the_self_shadow_is_read_at_the_drawn_layer_with_the_rows_as_they_are() {
    let character = "#version 150\n\
vec4 tex2D( sampler2D sampler, vec2 texcoord ) { return texture( sampler, texcoord.xy ); }\n\
vec2 screenPosToTexcoord( vec2 pos, vec4 scale ) { return pos * scale.xy + scale.zw; }\n\
uniform vec4 _fa_ [20];\n\
uniform sampler2D samp_selfshadowmap;\n\
in vec4 gl_FragCoord;\n\
out vec4 out_FragColor0;\n\
void main() {\n\
\tvec2 texcoord = screenPosToTexcoord( gl_FragCoord.xy, _fa_[17 ] );\n\
\tvec4 encoded = tex2D( samp_selfshadowmap, texcoord );\n\
\tout_FragColor0 = encoded;\n\
}\n";
    assert!(!reads_screen(character));
    let layered = eye_maps_fragment(character).expect("converts");
    assert_eq!(layered.maps, vec!["samp_selfshadowmap"]);
    assert_eq!(layered.rows, 0);
    let s = &layered.source;
    assert!(s.contains("vec4 encoded = ( gl_Layer != 0 ? tex2D( monaka_l_samp_selfshadowmap, texcoord ) : tex2D( samp_selfshadowmap, texcoord ) );"), "{s}");
    assert!(s.contains("uniform sampler2DArray monaka_l_samp_selfshadowmap;"), "{s}");
    // The rows as they were: the same for both eyes.
    assert!(s.contains("screenPosToTexcoord( gl_FragCoord.xy, _fa_[17 ] )") && !s.contains("monaka_fr") && !s.contains("monaka_fmain"), "{s}");
    // The scene's maps' conversion leaves it alone, and this one theirs.
    assert_eq!(layered_fragment(character), None);
    assert_eq!(eye_maps_fragment(LIGHT), None);
}

#[test]
fn every_shader_reading_the_self_shadow_converts() {
    let Ok(dir) = std::fs::read_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../research/renderprogs/converted")) else { return };
    let mut failed = Vec::new();
    let mut converted = 0;
    for entry in dir.flatten() {
        let path = entry.path();
        if path.extension().is_none_or(|e| e != "frag") {
            continue;
        }
        // As the engine has them: without the locations the research copies were given.
        let text: String = std::fs::read_to_string(&path).expect("reads").lines().map(|line| match line.strip_prefix("layout(location = ") {
            Some(rest) => format!("{}
", rest.split_once(") ").map_or(rest, |(_, after)| after)),
            None => format!("{line}
"),
        }).collect();
        if !text.contains("samp_selfshadowmap") || reads_screen(&text) {
            continue;
        }
        match eye_maps_fragment(&text) {
            Some(_) => converted += 1,
            None => failed.push(path.file_name().unwrap().to_string_lossy().into_owned()),
        }
    }
    assert!(failed.is_empty(), "{converted} converted; not: {failed:?}");
}
