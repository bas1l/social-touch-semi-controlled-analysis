import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np

# --- Standalone Visualization Function ---

def visualize_point_cloud_comparison(
    pcd_left: o3d.geometry.Geometry,
    pcd_right: o3d.geometry.Geometry,
    title: str = "Point Cloud Comparison",
    left_label: str = "Original",
    right_label: str = "Modified"
) -> None:
    """
    Launches a standalone Open3D GUI window with split-screen synchronized views
    and controls to recenter or flip the visualization.
    """
    print(f"👀 Interactive Mode: Launching Split-Screen Synchronized View.")

    app = gui.Application.instance
    try:
        app.initialize()
    except Exception as e:
        print(f"Open3D Application initialization notice: {e}")

    window = gui.Application.instance.create_window(title, 1280, 768)

    em = window.theme.font_size
    panel = gui.Horiz(0.5 * em, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))

    btn_recenter = gui.Button("Recenter Origin")
    btn_recenter.horizontal_padding_em = 0.5
    btn_recenter.vertical_padding_em = 0

    btn_flip = gui.Button("Flip Direction")
    btn_flip.horizontal_padding_em = 0.5
    btn_flip.vertical_padding_em = 0

    panel.add_child(btn_recenter)
    panel.add_child(btn_flip)

    widget_left = gui.SceneWidget()
    widget_left.scene = rendering.Open3DScene(window.renderer)
    widget_left.scene.set_background([0.1, 0.1, 0.1, 1.0])

    widget_right = gui.SceneWidget()
    widget_right.scene = rendering.Open3DScene(window.renderer)
    widget_right.scene.set_background([0.1, 0.1, 0.1, 1.0])

    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.point_size = 3.0

    widget_left.scene.add_geometry(left_label, pcd_left, mat)
    widget_right.scene.add_geometry(right_label, pcd_right, mat)

    bbox = pcd_left.get_axis_aligned_bounding_box()
    bbox_center = bbox.get_center()

    def on_recenter_click():
        widget_left.setup_camera(60.0, bbox, bbox_center)
        widget_left.force_redraw()

    def on_flip_click():
        model_matrix = widget_left.scene.camera.get_model_matrix()
        eye = model_matrix[:3, 3]
        R = model_matrix[:3, :3]
        up = R[:, 1]
        center_to_eye = eye - bbox_center
        new_eye = bbox_center - center_to_eye
        widget_left.scene.camera.look_at(bbox_center, new_eye, up)
        widget_left.force_redraw()

    btn_recenter.set_on_clicked(on_recenter_click)
    btn_flip.set_on_clicked(on_flip_click)

    def on_layout(layout_context):
        r = window.content_rect
        panel_height = panel.calc_preferred_size(layout_context, gui.Widget.Constraints()).height
        panel.frame = gui.Rect(r.x, r.y, r.width, panel_height)
        view_y = r.y + panel_height
        view_height = r.height - panel_height
        widget_left.frame = gui.Rect(r.x, view_y, r.width // 2, view_height)
        widget_right.frame = gui.Rect(r.x + r.width // 2, view_y, r.width // 2, view_height)

    window.set_on_layout(on_layout)
    window.add_child(panel)
    window.add_child(widget_left)
    window.add_child(widget_right)

    class CameraState:
        def __init__(self):
            self.left_matrix = np.eye(4)
            self.right_matrix = np.eye(4)

    state = CameraState()

    def apply_pose_to_camera(target_widget, source_matrix):
        eye = source_matrix[:3, 3]
        R = source_matrix[:3, :3]
        up = R[:, 1]
        forward = -R[:, 2]
        center = eye + forward
        target_widget.scene.camera.look_at(center, eye, up)

    def sync_loop():
        if widget_left.frame.height <= 0 or widget_right.frame.height <= 0:
            gui.Application.instance.post_to_main_thread(window, sync_loop)
            return

        current_left = widget_left.scene.camera.get_model_matrix()
        current_right = widget_right.scene.camera.get_model_matrix()

        if not np.allclose(current_left, state.left_matrix, atol=1e-6):
            apply_pose_to_camera(widget_right, current_left)

            cam_left = widget_left.scene.camera
            widget_right.scene.camera.set_projection(
                cam_left.get_field_of_view(),
                widget_right.frame.width / widget_right.frame.height,
                0.1, 1000.0,
                rendering.Camera.FovType.Vertical
            )
            widget_right.force_redraw()

            state.left_matrix = current_left
            state.right_matrix = widget_right.scene.camera.get_model_matrix()

        elif not np.allclose(current_right, state.right_matrix, atol=1e-6):
            apply_pose_to_camera(widget_left, current_right)

            cam_right = widget_right.scene.camera
            widget_left.scene.camera.set_projection(
                cam_right.get_field_of_view(),
                widget_left.frame.width / widget_left.frame.height,
                0.1, 1000.0,
                rendering.Camera.FovType.Vertical
            )
            widget_left.force_redraw()

            state.right_matrix = current_right
            state.left_matrix = widget_left.scene.camera.get_model_matrix()

        gui.Application.instance.post_to_main_thread(window, sync_loop)

    widget_left.setup_camera(60.0, bbox, bbox.get_center())
    gui.Application.instance.post_to_main_thread(window, sync_loop)
    gui.Application.instance.run()
