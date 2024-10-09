#![allow(warnings)]

pub use crate::rasterizer1::{Buffer, Primitive, Rasterizer};
pub use crate::shader::FragmentShaderPayload;
pub use crate::texture::Texture;
pub use crate::utils::*;
pub use nalgebra::Vector3;
pub use opencv::core::Vector;
use opencv::highgui::{imshow, wait_key};
use opencv::imgcodecs::imwrite;
pub use opencv::Result;
pub use std::env;

use std::io;

pub fn t1() -> Result<()> {
    println!("选择任务1");
    let mut angle = 0.0;

    let mut axis = V3f::new(0.0, 0.0, 0.0);
    let mut str = String::new();
    io::stdin().read_line(&mut str).expect("failed to read axis");
    let mut iter = str.trim().split_whitespace();
    axis.x = iter.next().expect("iterator failure")
        .parse().expect("failed to convert to int");
    axis.y = iter.next().expect("iterator failure")
        .parse().expect("failed to convert to int");
    axis.z = iter.next().expect("iterator failure")
        .parse().expect("failed to convert to int");

    println!("get x = {}, y = {}, z = {}", axis.x, axis.y, axis.z);
    
    let mut r = Rasterizer::new(700, 700);
    let eye_pos = Vector3::new(0.0, 0.0, 5.0);
    let pos = vec![
        Vector3::new(2.0, 0.0, -2.0),
        Vector3::new(0.0, 2.0, -2.0),
        Vector3::new(-2.0, 0.0, -2.0),
    ];
    let ind = vec![Vector3::new(0, 1, 2)];

    let pos_id = r.load_position(&pos);
    let ind_id = r.load_indices(&ind);

    let mut k = 0;
    let mut frame_count = 0;

    while k != 27 {
        r.clear(Buffer::Both);
        r.set_model(get_model_matrix_abri(angle, 1.0, axis));
        // r.set_model(get_model_matrix(angle, 1.0));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45.0, 1.0, 0.1, 50.0));
        r.draw_triangle(pos_id, ind_id, Primitive::Triangle);

        let frame_buffer = r.frame_buffer();
        let image = frame_buffer2cv_mat(frame_buffer);
        imshow("image", &image).unwrap();

        k = wait_key(80).unwrap();
        println!("frame count: {}", frame_count);
        if k == 'a' as i32 {
            angle += 10.0;
        } else if k == 'd' as i32 {
            angle -= 10.0;
        }
        frame_count += 1;
    }
    Ok(())
}
