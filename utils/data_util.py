import tensorflow as tf


def random_int(maxval, minval=0):
    return tf.random_uniform(
        shape=[], minval=minval, maxval=maxval, dtype=tf.int32)


def random_crop_and_resize(img, mask, height,
                           min_scale=0.9,
                           input_shape = (101, 101),
                           target_shape=(96, 96)):
    h, w = input_shape
    aspect_r = w / h
    h_min = int(min_scale * h)
    crop_h = random_int(maxval=h, minval=h_min)
    crop_w = tf.to_int32(aspect_r * tf.to_float(crop_h))
    top = random_int(h - crop_h)
    left = random_int(w - crop_w)
    new_img = img[top:top + crop_h, left:left + crop_w]
    new_mask = mask[top:top + crop_h, left:left + crop_w]
    # new_img = tf.random_crop(img, size=[crop_h, crop_w, 1])
    # new_mask = tf.random_crop(mask, size=[crop_h, crop_w, 1])
    h, w = target_shape
    # new_img = tf.expand_dims(new_img, axis=2)
    # new_mask = tf.expand_dims(new_mask, axis=2)
    new_img = tf.image.resize_images(new_img, size=[h, w])
    new_mask = tf.image.resize_images(new_mask, size=[h, w])
    return new_img, new_mask, height


def random_rotate(img, mask, height):
    k = random_int(3)
    # new_img = tf.expand_dims(img, axis=2)
    # new_mask = tf.expand_dims(mask, axis=2)
    new_img = tf.cond(k > 0,
                      true_fn=lambda: tf.image.rot90(img, k),
                      false_fn=lambda: img)
    new_mask = tf.cond(k > 0,
                       true_fn=lambda: tf.image.rot90(mask, k),
                       false_fn=lambda: mask)
    return new_img, new_mask, height


def random_flip_left_right(img, mask, height):
    random_var = random_int(2)
    random_var = tf.cast(random_var, tf.bool)
    # new_img = tf.expand_dims(img, axis=2)
    # new_mask = tf.expand_dims(mask, axis=2)
    flipped_img = tf.cond(random_var,
                          true_fn=lambda: tf.image.flip_left_right(img),
                          false_fn=lambda: tf.identity(img))
    flipped_mask = tf.cond(random_var,
                           true_fn=lambda: tf.image.flip_left_right(mask),
                           false_fn=lambda: tf.identity(mask))
    return flipped_img, flipped_mask, height
