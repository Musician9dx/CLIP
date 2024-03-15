class SelfAttention(Layer):

    def __init__(self, embedSize):
        super().__init__()
        self.embedSize = embedSize
        self.key = Dense(self.embedSize, activation="linear")
        self.query = Dense(self.embedSize, activation="linear")
        self.ad = Add()

    def call(self, logits):
        inputs = logits

        key = self.key(logits)
        query = self.query(logits)

        key = tf.matmul(query, key, transpose_b=True)

        value = tf.matmul(key, query)
        value = softmax(value)

        return value


class CrossAttention(Layer):

    def __init__(self, embedSize):
        super().__init__()
        self.embedSize = embedSize
        self.key = Dense(self.embedSize, activation="linear")
        self.query = Dense(self.embedSize, activation="linear")
        self.ad = Add()

    def call(self, logits1, logits2):
        key = self.key(logits1)
        query = self.query(logits2)

        key = tf.matmul(query, key, transpose_b=True)
        value = tf.matmul(key, query)
        value = softmax(value)

        return value


class MultiHeadAttention(Layer):

    def __init__(self, numHeads, embedSize):
        super().__init__()

        headSize = embedSize // numHeads

        self.attentionBlocks = [SelfAttention(headSize) for i in range(numHeads)]

    def call(self, logits):
        c = []

        for layer in self.attentionBlocks:
            op1 = layer(logits)
            c.append(op1)

        op2 = tf.concat(c, axis=-1)

        return op2


class Encoder(Layer):

    def __init__(self, vocabSize, embedSize):
        super().__init__()

        self.embedSize = embedSize
        self.conv = Conv2D(4, kernel_size=(3, 3), padding="same")
        self.d1 = Dense(100, activation="linear")
        self.mha = MultiHeadAttention(10, 100)
        self.ad = Add()
        self.pEmbedding=PositionEmbedding(vocabSize,embedSize)

    def build(self):
        self.self_attention = [SelfAttention(self.embedSize) for i in range(5)]

    def call(self, image):
        logits = self.conv(image)

        # Assuming 'logits' has shape (batch_size, height, width, channels)
        batch_size, height, width, channels = logits.shape

        # Size of the patches
        patch_size = 5

        # Number of patches along the height and width
        num_patches_height = height // patch_size
        num_patches_width = width // patch_size

        # Extract patches using tf.image.extract_patches
        patches = tf.image.extract_patches(
            images=logits,
            sizes=[1, patch_size, patch_size, 1],
            strides=[1, patch_size, patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )

        patches = tf.reshape(patches, (1, 5, 500))
        patches=patches+self.pEmbedding(patches)
        logits = self.mha(patches)

        for layer in self.self_attention:
            lg = logits
            logits = layer(logits)
            logits = self.ad([relu(logits), lg])

        return logits


class Decoder(Layer):

    def __init__(self, vocabSize, embedSize):
        super().__init__()

        self.vocabSize = vocabSize
        self.embedSize = embedSize
        self.embedding = Embedding(self.vocabSize, self.embedSize)
        self.self_attention = SelfAttention(self.embedSize)
        self.mha = MultiHeadAttention(4, 100)
        self.maskMatrix = tf.ones((self.embedSize, self.embedSize))
        self.maskMatrix = tf.linalg.band_part(self.maskMatrix, -1, 0)
        self.pEmbedding=PositionEmbedding(vocabSize,embedSize)

    def call(self, tokens):
        embedding = self.embedding(tokens)
        embedding=embedding+self.pEmbedding(patches)
        logits = self.mha(embedding)
        logits = self.self_attention(logits)
        logits = tf.matmul(logits, self.maskMatrix)
        return logits


class CLIP(Model):

    def __init__(self, vocabSize=5, embedSize=100):
        super().__init__()

        self.enc = Encoder(vocabSize, embedSize)
        self.dec = Decoder(vocabSize, embedSize)
        self.ca = CrossAttention(embedSize)
        self.layerAttention = [SelfAttention(embedSize) for i in range(4)]
        self.ad = Add()
        self.d1 = [Dense(embedSize, activation="relu") for i in range(5)]

    def call(self, encInputs, decInputs, targets=None):

        logits1 = self.enc(encInputs)
        logits2 = self.dec(decInputs)

        logits = self.ca(logits1, logits2)

        for layer in self.layerAttention:
            lg1 = logits
            logits = layer(logits)
            logits = self.ad([relu(lg1), logits])

        for layer in self.d1:
            logits = layer(logits)

        if targets is None:
            return logits, None
        else:
            B, T, C = tf.shape(logits)
            logits = tf.reshape(logits, (B * T, C))
            targets = tf.reshape(targets, (B * T,))
            loss = tf.keras.losses.sparse_categorical_crossentropy(targets, logits, from_logits=True)
            loss = tf.reduce_mean(loss)
            return logits, loss

    #     def generate(self, idx,new_tokens):
    #         for _ in range(new_tokens):
    #             logits, _ = self.call(idx)
    #             logits = logits[:, -1, :]
    #             probs = softmax(logits, axis=-1)
    #             idx_next = tf.reshape(tf.argmax(probs, axis=-1), (-1, 1))
    #             idx = tf.concat([idx, tf.cast(idx_next, tf.int32)], axis=1)
    #         return idx

    def fitM(self, xb, yb, targets, steps=100):
        optimizer = tf.keras.optimizers.Adam()

        for step in range(steps):
            with tf.GradientTape() as tape:
                logits, loss = self.call(xb, yb, targets)
                print(f"Step: {step}", float(loss))
            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))



