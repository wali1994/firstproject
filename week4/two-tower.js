/* two-tower.js — Two-Tower retrieval model for TF.js
 * - Shallow (embed•dot) and Deep (MLP towers with genres)
 * - Unique variable names per model instance (avoid collisions)
 */

class TwoTowerModel {
  constructor(numUsers, numItems, opts = {}) {
    this.numUsers = numUsers;
    this.numItems = numItems;

    // unique suffix so TF.js variable names never collide
    this.uid = `m${Date.now()}_${Math.floor(Math.random() * 1e9)}`;

    this.embeddingDim = opts.embeddingDim ?? 32;
    this.useDeep      = !!opts.useDeep;
    this.userHidden   = (opts.userHidden ?? [64, this.embeddingDim]).slice();
    this.itemHidden   = (opts.itemHidden ?? [64, this.embeddingDim]).slice();

    const uLast = this.userHidden.at(-1);
    const iLast = this.itemHidden.at(-1);
    if (this.useDeep && uLast !== iLast) {
      throw new Error(`userHidden last (${uLast}) must equal itemHidden last (${iLast})`);
    }
    this.towerOutDim = this.useDeep ? uLast : this.embeddingDim;

    this.numGenres   = opts.numGenres ?? 0;
    this.userFeatDim = opts.userFeatDim ?? 0;
    this.l2Reg       = opts.l2Reg ?? 0.0;
    this.lr          = opts.lr ?? 1e-3;

    // embeddings (unique names)
    this.userEmb = tf.variable(
      tf.randomNormal([numUsers, this.embeddingDim], 0, 0.05),
      true, `userEmb_${this.uid}`
    );
    this.itemEmb = tf.variable(
      tf.randomNormal([numItems, this.embeddingDim], 0, 0.05),
      true, `itemEmb_${this.uid}`
    );

    if (this.useDeep) {
      const uIn = this.embeddingDim + this.userFeatDim;
      this.userW = this._makeDenseStack('U', uIn, this.userHidden);
      const iIn = this.embeddingDim + this.numGenres;
      this.itemW = this._makeDenseStack('I', iIn, this.itemHidden);
    }

    this.optimizer = tf.train.adam(this.lr);
    this.itemIndex = null; // cached item tower vectors for scoring
  }

  _makeDenseStack(prefix, inDim, sizes) {
    const layers = [];
    let prev = inDim;
    sizes.forEach((units, k) => {
      const nameW = `${prefix}_${this.uid}_W_${k}`;
      const nameB = `${prefix}_${this.uid}_b_${k}`;
      layers.push({
        W: tf.variable(tf.randomNormal([prev, units], 0, Math.sqrt(2/(prev+units))), true, nameW),
        b: tf.variable(tf.zeros([units]), true, nameB)
      });
      prev = units;
    });
    return layers;
  }

  _forwardMLP(x, weights) {
    let h = x;
    for (let i = 0; i < weights.length; i++) {
      const { W, b } = weights[i];
      h = tf.add(tf.matMul(h, W), b);
      if (i < weights.length - 1) h = tf.relu(h);
    }
    return h;
  }

  _userTower(userIdx, userFeats = null) {
    const uEmb = tf.gather(this.userEmb, userIdx); // [B, emb]
    if (!this.useDeep) return uEmb;
    let uX = uEmb;
    if (this.userFeatDim > 0 && userFeats) uX = tf.concat([uX, userFeats], 1);
    return this._forwardMLP(uX, this.userW);      // [B, out]
  }

  _itemTower(itemIdx, itemGenres = null) {
    const iEmb = tf.gather(this.itemEmb, itemIdx); // [B, emb]
    if (!this.useDeep) return iEmb;
    let iX = iEmb;
    if (this.numGenres > 0 && itemGenres) iX = tf.concat([iX, itemGenres], 1);
    return this._forwardMLP(iX, this.itemW);      // [B, out]
  }

  _batchSoftmaxLoss(U, V) {
    const logits = tf.matMul(U, V, false, true); // [B,B]
    const labels = tf.oneHot(tf.range(0, logits.shape[0], 1, 'int32'), logits.shape[1]);
    const ce = tf.losses.softmaxCrossEntropy(labels, logits);
    if (this.l2Reg > 0) {
      const l2 = tf.add(tf.sum(tf.square(U)), tf.sum(tf.square(V)));
      return tf.add(ce, tf.mul(this.l2Reg, l2));
    }
    return ce;
  }

  async trainStep(userIdxArr, itemIdxArr, { userFeats=null, itemGenres=null } = {}) {
    const lossVal = await this.optimizer.minimize(() => {
      return tf.tidy(() => {
        const uIdx = tf.tensor1d(userIdxArr, 'int32');
        const iIdx = tf.tensor1d(itemIdxArr, 'int32');
        const uF = (this.userFeatDim > 0 && userFeats) ? tf.tensor2d(userFeats) : null;
        const iG = (this.numGenres > 0 && itemGenres) ? tf.tensor2d(itemGenres) : null;

        const U = this._userTower(uIdx, uF);
        const V = this._itemTower(iIdx, iG);
        const loss = this._batchSoftmaxLoss(U, V);

        uIdx.dispose(); iIdx.dispose();
        if (uF) uF.dispose();
        if (iG) iG.dispose();
        return loss;
      });
    }, true);

    const scalar = (await lossVal.data())[0];
    lossVal.dispose?.();
    return scalar;
  }

  buildItemIndex(allItemGenres = null) {
    tf.tidy(() => {
      let V = this.itemEmb;
      if (this.useDeep) {
        let x = V;
        if (this.numGenres > 0) {
          if (!allItemGenres) throw new Error('buildItemIndex requires item genres');
          x = tf.concat([x, allItemGenres], 1);
        }
        for (let i = 0; i < this.itemW.length; i++) {
          const { W, b } = this.itemW[i];
          x = tf.add(tf.matMul(x, W), b);
          if (i < this.itemW.length - 1) x = tf.relu(x);
        }
        V = x;
      }
      if (this.itemIndex) this.itemIndex.dispose();
      this.itemIndex = V.clone();
    });
  }

  async scoreAllForUser(userId, userFeatRow = null) {
    if (!this.itemIndex) throw new Error('Call buildItemIndex() after training');
    const scoresTensor = tf.tidy(() => {
      const idx = tf.tensor1d([userId], 'int32');
      let u = tf.gather(this.userEmb, idx);
      if (this.useDeep) {
        if (this.userFeatDim > 0 && userFeatRow) u = tf.concat([u, userFeatRow], 1);
        for (let i = 0; i < this.userW.length; i++) {
          const { W, b } = this.userW[i];
          u = tf.add(tf.matMul(u, W), b);
          if (i < this.userW.length - 1) u = tf.relu(u);
        }
      }
      return tf.matMul(u, this.itemIndex, false, true); // [1, N]
    });
    const scores = Array.from(await scoresTensor.data());
    scoresTensor.dispose();
    const topIdx = scores.map((s, i) => [s, i]).sort((a,b)=>b[0]-a[0]).map(x=>x[1]);
    return { scores, topIdx };
  }

  getItemVectors() {
    if (!this.itemIndex) throw new Error('Call buildItemIndex() first');
    return this.itemIndex;
  }

  dispose() {
    this.userEmb.dispose();
    this.itemEmb.dispose();
    if (this.userW) this.userW.forEach(({ W, b }) => { W.dispose(); b.dispose(); });
    if (this.itemW) this.itemW.forEach(({ W, b }) => { W.dispose(); b.dispose(); });
    if (this.itemIndex) this.itemIndex.dispose();
  }
}

window.TwoTowerModel = TwoTowerModel;
