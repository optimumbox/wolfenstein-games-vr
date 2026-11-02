use super::*;

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
    assert!(out.starts_with("#version 150\n#extension GL_NV_viewport_array2 : require\n#extension GL_NV_stereo_view_rendering : require\n"));
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
